import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

try:
    import numpy as np
    import sentencepiece as spm
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import GPT2Config, GPT2LMHeadModel, get_cosine_schedule_with_warmup
except ModuleNotFoundError as exc:
    missing_name = exc.name
    print(
        f"Missing dependency: {missing_name}\n"
        "Install pretraining dependencies with:\n"
        "  pip install -r requirements_pretrain.txt",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


# 当前脚本所在目录。默认相对路径都会基于项目根目录解析。
BASE_DIR = Path(__file__).resolve().parent


def resolve_path(path: str) -> Path:
    """把命令行传入的路径转成绝对路径，方便从任意工作目录运行脚本。"""
    path = Path(path)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def count_parameters(model: torch.nn.Module) -> int:
    """统计可训练参数量，用来确认模型规模是否接近 0.1B。"""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_token_cache(args, sp: spm.SentencePieceProcessor, token_cache_path: Path) -> Path:
    """
    将清洗后的 wiki 文本预先编码成连续 token id，并保存为 .npy。

    预训练时反复从磁盘读取原始文本再分词会很慢，所以第一次运行先做缓存；
    后续如果语料和 SentencePiece 模型没有变化，可以直接复用这个缓存。
    """
    corpus_path = resolve_path(args.corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    token_cache_path.parent.mkdir(parents=True, exist_ok=True)
    eos_id = sp.eos_id()
    if eos_id < 0:
        raise ValueError("SentencePiece model must define eos_id. Your clean script uses eos_id=3.")

    token_ids = []
    docs_seen = 0
    start_time = time.time()

    with corpus_path.open("r", encoding="utf-8") as fin:
        buffer = []
        for line in fin:
            stripped = line.strip()

            # clean_sentencepiece.py 用空行分隔文档，这里按文档追加 eos，帮助模型学习文章边界。
            if stripped:
                buffer.append(stripped)
                continue

            if buffer:
                text = "\n".join(buffer)
                token_ids.extend(sp.encode(text, out_type=int))
                token_ids.append(eos_id)
                docs_seen += 1
                buffer = []

            if args.max_token_cache_tokens and len(token_ids) >= args.max_token_cache_tokens:
                token_ids = token_ids[: args.max_token_cache_tokens]
                break

        # 处理文件末尾没有空行的最后一篇文档。
        if buffer and (not args.max_token_cache_tokens or len(token_ids) < args.max_token_cache_tokens):
            text = "\n".join(buffer)
            token_ids.extend(sp.encode(text, out_type=int))
            token_ids.append(eos_id)
            docs_seen += 1

    if not token_ids:
        raise ValueError(f"No tokens were produced from corpus: {corpus_path}")

    # 你的词表大小默认是 32000，uint16 足够保存 token id，缓存体积更小。
    token_array = np.asarray(token_ids, dtype=np.uint16)
    np.save(token_cache_path, token_array)

    stats = {
        "corpus_path": str(corpus_path),
        "spm_model": str(resolve_path(args.spm_model)),
        "docs_seen": docs_seen,
        "tokens": int(token_array.size),
        "dtype": str(token_array.dtype),
        "seconds": round(time.time() - start_time, 2),
    }
    with token_cache_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"[cache] saved {token_array.size:,} tokens from {docs_seen:,} docs to {token_cache_path}")
    return token_cache_path


class CausalBlockDataset(Dataset):
    """把连续 token 流切成固定长度训练样本，用于 GPT 自回归预训练。"""

    def __init__(self, token_ids: np.ndarray, block_size: int, start: int, end: int):
        if end - start <= block_size:
            raise ValueError(
                f"Not enough tokens for block_size={block_size}. "
                f"Available tokens in split: {end - start}."
            )
        self.token_ids = token_ids
        self.block_size = block_size
        self.start = start
        self.end = end
        self.length = end - start - block_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        offset = self.start + index
        chunk = self.token_ids[offset : offset + self.block_size + 1].astype(np.int64)

        # GPT2LMHeadModel 会在内部把 labels 右移一位，所以 labels 应该和 input_ids 相同。
        # 如果这里手动使用 chunk[1:]，就会变成“错开两位”的训练目标。
        input_ids = torch.from_numpy(chunk[:-1])
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


def build_model_config(args, vocab_size: int, pad_id: int, bos_id: int, eos_id: int) -> GPT2Config:
    """
    构建约 0.1B 参数的 GPT2 配置。

    默认配置：10 层、隐藏维度 768、12 个注意力头，词表 32k 时大约 96M 可训练参数。
    如果你想更接近 110M，可以把 --n_layer 调到 12。
    """
    return GPT2Config(
        vocab_size=vocab_size,
        n_positions=args.block_size,
        n_ctx=args.block_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=args.n_inner,
        activation_function="gelu_new",
        resid_pdrop=args.dropout,
        embd_pdrop=args.dropout,
        attn_pdrop=args.dropout,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        use_cache=False,
    )


def save_checkpoint(output_dir: Path, model: GPT2LMHeadModel, optimizer, scheduler, step: int, epoch: int):
    """保存可继续训练的 checkpoint，同时保留 transformers 标准模型文件。"""
    checkpoint_dir = output_dir / f"checkpoint-step-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        checkpoint_dir / "trainer_state.pt",
    )
    print(f"[save] checkpoint written to {checkpoint_dir}")


@torch.no_grad()
def evaluate(model: GPT2LMHeadModel, dataloader: DataLoader, device: torch.device, max_batches: int) -> float:
    """在验证集上计算平均 loss，便于观察训练是否真的在下降。"""
    model.eval()
    losses = []
    for batch_index, batch in enumerate(dataloader, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        losses.append(outputs.loss.detach().float().item())
        if max_batches and batch_index >= max_batches:
            break
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain a ~0.1B Chinese GPT model with SentencePiece tokens.")

    # 输入输出路径：默认直接接你的 clean_sentencepiece 产物。
    parser.add_argument("--corpus_path", default="clean_sentencepiece/wiki_zh_clean.txt")
    parser.add_argument("--spm_model", default="clean_sentencepiece/chinese_wiki_spm_bpe.model")
    parser.add_argument("--output_dir", default="pretrain_gpt2_0_1b")
    parser.add_argument("--token_cache", default="pretrain_gpt2_0_1b/wiki_zh_tokens.npy")
    parser.add_argument("--overwrite_cache", action="store_true", help="Rebuild token cache even if it exists.")

    # 模型大小：默认约 0.1B。词表来自 SentencePiece，不需要下载外部预训练模型。
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=10)
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--n_inner", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)

    # 训练参数：默认偏保守，适合先跑通；显存足够时可以增大 batch_size 或 block_size。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Use CUDA fp16 mixed precision.")

    # 日志、保存和调试参数。
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=0, help="Debug only: stop after N optimizer steps.")
    parser.add_argument(
        "--max_token_cache_tokens",
        type=int,
        default=0,
        help="Debug only: cache at most N tokens from the corpus.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spm_model_path = resolve_path(args.spm_model)
    if not spm_model_path.exists():
        raise FileNotFoundError(f"SentencePiece model not found: {spm_model_path}")

    sp = spm.SentencePieceProcessor()
    sp.load(str(spm_model_path))
    vocab_size = sp.get_piece_size()

    token_cache_path = resolve_path(args.token_cache)
    if args.overwrite_cache or not token_cache_path.exists():
        build_token_cache(args, sp, token_cache_path)
    else:
        print(f"[cache] using existing token cache: {token_cache_path}")

    token_ids = np.load(token_cache_path, mmap_mode="r")
    if int(token_ids.max()) >= vocab_size:
        raise ValueError("Token cache contains ids outside the SentencePiece vocabulary.")

    split_index = int(len(token_ids) * (1.0 - args.val_ratio))
    split_index = max(args.block_size + 1, min(split_index, len(token_ids) - args.block_size - 1))
    train_dataset = CausalBlockDataset(token_ids, args.block_size, 0, split_index)
    val_dataset = CausalBlockDataset(token_ids, args.block_size, split_index, len(token_ids))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    config = build_model_config(args, vocab_size, sp.pad_id(), sp.bos_id(), sp.eos_id())
    model = GPT2LMHeadModel(config)
    param_count = count_parameters(model)

    # 把模型配置和分词器文件一起放到输出目录，后续推理或继续训练更方便。
    model.save_pretrained(output_dir)
    shutil.copy2(spm_model_path, output_dir / spm_model_path.name)
    with (output_dir / "pretrain_args.json").open("w", encoding="utf-8") as f:
        payload = vars(args).copy()
        payload["parameter_count"] = param_count
        payload["vocab_size"] = vocab_size
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = args.max_steps if args.max_steps else steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    print(f"[model] parameters: {param_count / 1e6:.2f}M")
    print(f"[data] train blocks: {len(train_dataset):,}, val blocks: {len(val_dataset):,}")
    print(f"[train] device={device}, total_steps={total_steps}, warmup_steps={warmup_steps}")

    global_step = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        for micro_step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}

            with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            running_loss += loss.detach().float().item()

            if micro_step % args.gradient_accumulation_steps != 0:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.logging_steps == 0:
                avg_loss = running_loss * args.gradient_accumulation_steps / args.logging_steps
                lr = scheduler.get_last_lr()[0]
                print(f"[step {global_step}] train_loss={avg_loss:.4f} lr={lr:.6g}")
                running_loss = 0.0

            if args.eval_steps and global_step % args.eval_steps == 0:
                val_loss = evaluate(model, val_loader, device, max_batches=50)
                ppl = math.exp(min(val_loss, 20.0))
                print(f"[eval {global_step}] val_loss={val_loss:.4f} ppl={ppl:.2f}")

            if args.save_steps and global_step % args.save_steps == 0:
                save_checkpoint(output_dir, model, optimizer, scheduler, global_step, epoch)

            if args.max_steps and global_step >= args.max_steps:
                break

        if args.max_steps and global_step >= args.max_steps:
            break

    model.save_pretrained(output_dir)
    print(f"[done] final model saved to {output_dir}")


if __name__ == "__main__":
    main()
 