import argparse
import json
import re
import unicodedata
from pathlib import Path


# 当前脚本所在目录。所有默认相对路径都会基于这个目录解析，
# 这样无论你从项目根目录还是脚本目录运行，都不容易出现找不到文件的问题。
BASE_DIR = Path(__file__).resolve().parent


def resolve_path(path: str) -> Path:
    """把命令行传入的路径转换成绝对路径。"""
    path = Path(path)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def iter_wiki_files(wiki_dir: Path):
    """遍历 WikiExtractor 产生的 wiki_* 分片文件。"""
    return sorted(path for path in wiki_dir.rglob("wiki_*") if path.is_file())


def maybe_fix_mojibake(text: str) -> str:
    """
    可选的乱码修复。

    有些中文文本可能是 UTF-8 字节被错误地按 GBK/CP936 解码后得到的。
    当前工作区中的 wiki 文件用 Python 读取是正常中文，所以默认不启用。
    只有你明确传入 --fix_mojibake 时，clean_text 才会调用这个函数。
    """
    try:
        fixed = text.encode("gbk").decode("utf-8")
    except UnicodeError:
        return text

    # 如果修复后这些常见乱码片段明显变少，就采用修复后的文本。
    bad_markers = ("鏁", "涓", "鐨", "鍥", "瑙", "绋", "€")
    if sum(text.count(marker) for marker in bad_markers) > sum(fixed.count(marker) for marker in bad_markers):
        return fixed
    return text


def clean_text(text: str, fix_mojibake: bool = False) -> str:
    """对单篇文章标题或正文做基础清洗。"""
    if fix_mojibake:
        text = maybe_fix_mojibake(text)

    # NFKC 会把全角英文、全角数字等规范化为更统一的形式。
    text = unicodedata.normalize("NFKC", text)

    # 统一 Windows / Unix / 老 Mac 的换行符。
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 全角空格替换成普通空格。
    text = text.replace("\u3000", " ")

    # 去掉零宽字符、BOM 和方向控制符，这些字符对语言建模通常没有帮助。
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", text)

    cleaned_chars = []
    for char in text:
        # 保留换行和制表符，后面再统一整理空白。
        if char == "\n" or char == "\t":
            cleaned_chars.append(char)
        elif char.isprintable():
            cleaned_chars.append(char)
        else:
            cleaned_chars.append(" ")
    text = "".join(cleaned_chars)

    # 去掉残留 URL，避免 tokenizer 学到大量低价值链接碎片。
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 合并连续空格，并整理换行两侧多余空格。
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)

    # 三个以上连续换行压缩成两个，保留文章段落感。
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 去掉空段落。
    paragraphs = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if paragraph:
            paragraphs.append(paragraph)
    return "\n".join(paragraphs)


def looks_useful(text: str, min_chars: int, min_chinese_ratio: float) -> bool:
    """过滤太短或中文比例过低的文章。"""
    if len(text) < min_chars:
        return False
    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    return chinese_count / max(len(text), 1) >= min_chinese_ratio


def extract_wiki_corpus(args):
    """从整个 wiki_zh 目录提取清洗后的纯文本训练语料。"""
    wiki_dir = resolve_path(args.wiki_dir)
    output_dir = resolve_path(args.output_dir)
    corpus_path = resolve_path(args.corpus_path)
    stats_path = output_dir / "clean_stats.json"
    sample_path = output_dir / "clean_sample.txt"

    # 创建输出目录。
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)

    # 收集全部 wiki_* 分片。调试时可以用 --max_files 限制文件数量。
    wiki_files = iter_wiki_files(wiki_dir)
    if args.max_files:
        wiki_files = wiki_files[: args.max_files]
    if not wiki_files:
        raise FileNotFoundError(f"No wiki_* files found under {wiki_dir}")

    stats = {
        "wiki_dir": str(wiki_dir),
        "files_seen": len(wiki_files),
        "lines_seen": 0,
        "json_errors": 0,
        "docs_written": 0,
        "docs_skipped": 0,
        "characters_written": 0,
    }

    # 用 id 做轻量去重。维基分片一般不会重复，但保留这个检查更稳。
    seen_ids = set()
    with corpus_path.open("w", encoding="utf-8", newline="\n") as fout, sample_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as fsample:
        for file_index, wiki_file in enumerate(wiki_files, start=1):
            with wiki_file.open("r", encoding="utf-8") as fin:
                for line in fin:
                    stats["lines_seen"] += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        stats["json_errors"] += 1
                        continue

                    doc_id = obj.get("id")
                    if doc_id and doc_id in seen_ids:
                        stats["docs_skipped"] += 1
                        continue
                    if doc_id:
                        seen_ids.add(doc_id)

                    # WikiExtractor 每行通常有 title 和 text 字段。
                    title = clean_text(obj.get("title", ""), fix_mojibake=args.fix_mojibake)
                    text = clean_text(obj.get("text", ""), fix_mojibake=args.fix_mojibake)

                    # 如果正文没有以标题开头，就把标题补到最前面。
                    # 这能让模型学习标题与正文的自然关联。
                    if title and not text.startswith(title):
                        text = title + "\n" + text

                    # 过滤太短、中文比例太低的文档。
                    if not looks_useful(text, args.min_chars, args.min_chinese_ratio):
                        stats["docs_skipped"] += 1
                        continue

                    # 每篇文章之间空一行。SentencePiece 会按行读取，保留这种边界比较清楚。
                    fout.write(text)
                    fout.write("\n\n")
                    stats["docs_written"] += 1
                    stats["characters_written"] += len(text)

                    # 写少量样本，方便人工快速检查清洗质量。
                    if stats["docs_written"] <= args.sample_docs:
                        fsample.write("=" * 80 + "\n")
                        fsample.write(text[: args.sample_chars])
                        fsample.write("\n\n")

                    # 调试时可以用 --max_docs 提前停止。
                    if args.max_docs and stats["docs_written"] >= args.max_docs:
                        break
            print(f"[extract] {file_index}/{len(wiki_files)} {wiki_file} docs={stats['docs_written']}")
            if args.max_docs and stats["docs_written"] >= args.max_docs:
                break

    # 保存清洗统计，便于判断跳过了多少文档、是否有 JSON 错误。
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Clean corpus: {corpus_path}")
    print(f"Sample file: {sample_path}")
    print(f"Stats file: {stats_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return corpus_path


def train_sentencepiece(args, corpus_path: Path):
    """使用清洗后的纯文本训练 SentencePiece tokenizer。"""
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise SystemExit("sentencepiece is not installed. Install it with: pip install sentencepiece") from exc

    model_prefix = resolve_path(args.model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    # 这里默认训练 BPE tokenizer。
    # pad/unk/bos/eos 使用固定 id，后续训练模型时更容易对齐特殊 token。
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        user_defined_symbols=["<mask>", "<sep>", "<cls>"],
        # 对大语料，抽样训练 tokenizer 会快很多，也足够稳定。
        # 如果想用全部句子训练，可传 --input_sentence_size 0。
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=True,
        # False 表示如果真实可用词表略小于指定 vocab_size，不直接报错。
        hard_vocab_limit=False,
    )
    print(f"SentencePiece model: {model_prefix}.model")
    print(f"SentencePiece vocab: {model_prefix}.vocab")


def parse_args():
    """命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Clean all wiki_zh JSONL files and train a SentencePiece tokenizer."
    )
    parser.add_argument("--wiki_dir", default="wiki_zh", help="Directory containing wiki_zh/AA/wiki_00 style files.")
    parser.add_argument("--output_dir", default="clean_sentencepiece", help="Directory for cleaned corpus and logs.")
    parser.add_argument("--corpus_path", default="clean_sentencepiece/wiki_zh_clean.txt")
    parser.add_argument("--model_prefix", default="clean_sentencepiece/chinese_wiki_spm_bpe")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--model_type", default="bpe", choices=["bpe", "unigram"])
    parser.add_argument("--character_coverage", type=float, default=0.9995)
    parser.add_argument(
        "--input_sentence_size",
        type=int,
        default=2000000,
        help="SentencePiece training sentence sample size. Use 0 for all sentences.",
    )
    parser.add_argument("--min_chars", type=int, default=80)
    parser.add_argument("--min_chinese_ratio", type=float, default=0.25)
    parser.add_argument("--max_files", type=int, default=0, help="Debug only: process at most N wiki files.")
    parser.add_argument("--max_docs", type=int, default=0, help="Debug only: write at most N documents.")
    parser.add_argument("--sample_docs", type=int, default=5)
    parser.add_argument("--sample_chars", type=int, default=1200)
    parser.add_argument("--fix_mojibake", action="store_true", help="Try to fix UTF-8 text decoded as GBK.")
    parser.add_argument("--skip_train", action="store_true", help="Only clean corpus; do not train SentencePiece.")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    cleaned_corpus_path = extract_wiki_corpus(parsed_args)
    if not parsed_args.skip_train:
        train_sentencepiece(parsed_args, cleaned_corpus_path)
