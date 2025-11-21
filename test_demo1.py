from FlagEmbedding import BGEM3FlagModel
import numpy as np

# 1️⃣ 加载模型
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# 2️⃣ 假设这是你题库中的已有题目
question_bank = [
    "What is the capital of France?",
    "Explain Newton's first law of motion.",
    "Define photosynthesis process.",
    "What is the main function of the CPU?"
]

# 3️⃣ 对题库题目进行编码（仅需做一次，可缓存）
bank_embeddings = model.encode(question_bank, return_dense=True, return_sparse=True, return_colbert_vecs=True)

# 4️⃣ 老师输入的新题目
new_question = "What is the role of the CPU in a computer?"

# 5️⃣ 编码新题
new_emb = model.encode([new_question], return_dense=True, return_sparse=True, return_colbert_vecs=True)

# 6️⃣ 计算相似度分数（融合dense+colbert）
pairs = [[new_question, q] for q in question_bank]
scores = model.compute_score(pairs, weights_for_different_modes=[0.5, 0, 0.5])["colbert+sparse+dense"]

# 7️⃣ 设定相似度阈值，筛选可能重复题
threshold = 0.75
for q, score in zip(question_bank, scores):
    if score >= threshold:
        print(f"⚠️ 疑似重复题：相似度 {score:.3f}")
        print(f"原题：{q}")
