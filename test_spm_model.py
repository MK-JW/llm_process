import sentencepiece as spm

# 你的模型路径
model_path = "D:\Desktop\llm_project\clean_sentencepiece_spm_test\chinese_wiki_spm_bpe.model"

# 加载 SentencePiece 模型
sp = spm.SentencePieceProcessor()
sp.load(model_path)

print("=== 基本信息 ===")
print("词表大小:", sp.get_piece_size())
print("pad id:", sp.pad_id(), sp.id_to_piece(sp.pad_id()))
print("unk id:", sp.unk_id(), sp.id_to_piece(sp.unk_id()))
print("bos id:", sp.bos_id(), sp.id_to_piece(sp.bos_id()))
print("eos id:", sp.eos_id(), sp.id_to_piece(sp.eos_id()))

# 测试文本
texts = [
    "人工智能正在改变世界，你是老逼。",
]

for text in texts:
    print("\n=== 测试文本 ===")
    print("原文:", text)

    # 编码成 token/piece
    pieces = sp.encode(text, out_type=str)
    print("token/piece:", pieces)

    # 编码成 token ID
    ids = sp.encode(text, out_type=int)
    print("token IDs:", ids)

    # 解码回文本
    decoded = sp.decode(ids)
    print("解码回文本:", decoded)