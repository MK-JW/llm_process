import argparse
import sys
from pathlib import Path

try:
    import sentencepiece as spm
    import torch
    from transformers import GPT2LMHeadModel
except ModuleNotFoundError as exc:
    print(
        f"Missing dependency: {exc.name}\n"
        "Install inference dependencies with:\n"
        "  pip install -r requirements_pretrain.txt",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


# 当前脚本所在目录。默认相对路径都会基于项目根目录解析。
BASE_DIR = Path(__file__).resolve().parent


def resolve_path(path: str) -> Path:
    """把命令行传入的路径转成绝对路径。"""
    path = Path(path)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def parse_args():
    parser = argparse.ArgumentParser(description="Run local GPT2 + SentencePiece inference on CPU.")
    parser.add_argument("--model_dir", default="pretrain_gpt2_0_1b", help="Directory containing config/model files.")
    parser.add_argument(
        "--spm_model",
        default="pretrain_gpt2_0_1b/chinese_wiki_spm_bpe.model",
        help="SentencePiece model used during pretraining.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="If set, run one generation and exit. If empty, enter interactive mode.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=80, help="Number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature. Lower is safer.")
    parser.add_argument("--top_k", type=int, default=50, help="Keep only top-k tokens while sampling.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling threshold.")
    parser.add_argument("--do_sample", action="store_true", default=True, help="Use sampling instead of greedy decoding.")
    parser.add_argument("--greedy", action="store_true", help="Disable sampling and use greedy decoding.")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Penalty for repeated tokens.")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3, help="Avoid repeating n-grams of this size.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_spm(spm_model_path: Path) -> spm.SentencePieceProcessor:
    """加载 SentencePiece，用它在中文文本和 token id 之间转换。"""
    processor = spm.SentencePieceProcessor()
    processor.load(str(spm_model_path))
    return processor


def get_device() -> torch.device:
    """优先使用 GPU；没有 CUDA 时自动退回 CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir: Path, device: torch.device) -> GPT2LMHeadModel:
    """加载本地训练好的 GPT2 模型，并放到当前可用设备上。"""
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model


def generate_reply(
    model: GPT2LMHeadModel,
    sp: spm.SentencePieceProcessor,
    prompt: str,
    args,
    device: torch.device,
) -> str:
    """根据一段输入文本生成续写结果。"""
    input_ids = sp.encode(prompt, out_type=int)
    if not input_ids:
        return "输入为空，无法生成。"

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    generate_kwargs = {
        "input_ids": input_tensor,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample and not args.greedy,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "pad_token_id": sp.pad_id(),
        "eos_token_id": sp.eos_id(),
        "bos_token_id": sp.bos_id(),
    }
    if generate_kwargs["do_sample"]:
        generate_kwargs.update(
            {
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
            }
        )

    # 推理不需要梯度，关闭后 CPU 会更省内存。
    with torch.no_grad():
        output_ids = model.generate(**generate_kwargs)

    return sp.decode(output_ids[0].tolist())


def run_once(model: GPT2LMHeadModel, sp: spm.SentencePieceProcessor, args, device: torch.device):
    """命令行传入 --prompt 时，只生成一次并退出。"""
    generated_text = generate_reply(model, sp, args.prompt, args, device)
    print("=== Prompt ===")
    print(args.prompt)
    print("\n=== Generated ===")
    print(generated_text)


def run_interactive(model: GPT2LMHeadModel, sp: spm.SentencePieceProcessor, args, device: torch.device):
    """交互式问答模式：每次输入一句，模型生成一次。"""
    print("进入交互式推理模式。输入 exit、quit 或 q 退出。")
    print("提示：当前模型只训练了很少 step，回答可能会很随机。")
    while True:
        try:
            prompt = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if prompt.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            break
        if not prompt:
            continue

        generated_text = generate_reply(model, sp, prompt, args, device)
        print(f"模型：{generated_text}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model_dir = resolve_path(args.model_dir)
    spm_model_path = resolve_path(args.spm_model)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not spm_model_path.exists():
        raise FileNotFoundError(f"SentencePiece model not found: {spm_model_path}")

    device = get_device()
    print(f"Using device: {device}")
    sp = load_spm(spm_model_path)
    model = load_model(model_dir, device)

    if args.prompt:
        run_once(model, sp, args, device)
    else:
        run_interactive(model, sp, args, device)


if __name__ == "__main__":
    main()
