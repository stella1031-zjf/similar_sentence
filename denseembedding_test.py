from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3',
                       use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
'''
sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]
# [[0.6265, 0.3477], [0.3499, 0.678 ]]

sentences_1 = [
    "以下哪项不是出自《楚辞》的作品？",
    "ENIAC是世界上第一台计算机。",
    "民事法律关系的主体包括自然人和法人。",
    "专升本考试一般在每年的几月份举行？"
]

sentences_2 = [
    "《九章》《九歌》《九辩》均出自《楚辞》。",
    "ENIAC是第一台电子数字计算机。",
    "民事法律关系的主体包括自然人、法人和非法人组织。",
    "专升本考试通常在每年四月左右举行。"
]
[[0.699  0.2524 0.2351 0.1926]
 [0.3564 0.9077 0.319  0.3042]
 [0.2925 0.3154 0.978  0.3088]
 [0.2715 0.2372 0.2556 0.867 ]]


sentences_1 = [
    "《诗经》的艺术特色是什么？",
    "《离骚》是谁的作品？",
    "《红楼梦》的思想主题是什么？",
    "屈原是哪一时期的诗人？"
]

sentences_2 = [
    "《诗经》的表现手法主要是赋、比、兴。",
    "屈原是战国时期的楚国诗人。",
    "《红楼梦》主要揭示封建社会的衰亡。",
    "《离骚》的作者是屈原。"
]


sentences_1 = [
    "求函数 f(x)=x^2+3x+2 的导数。",           # 导数
    "求极限 lim(x→0) sin(x)/x 的值。",          # 极限
    "计算 ∫₀¹ x² dx。",                        # 积分
    "判断矩阵是否可逆。",                       # 线代
    "求从 0 到 π/2 的 cos(x) 的定积分。",       # 积分
    "线性方程组有唯一解的条件是什么？",          # 线代
    "判断函数 f(x)=1/x 在 x=0 处是否连续。",     # 连续
    "已知 f(x)=ln(x)，求其导数。",               # 导数
]

sentences_2 = [
    "函数 f(x)=1/x 在 x=0 处是否有间断点？",     # 连续
    "当 x 无限增大时，(1+1/x)^x 的极限为 e。",   # 极限（干扰项）
    "∫₀¹ x² dx = 1/3。",                        # 积分
    "如果行列式不为 0，则矩阵可逆。",             # 线代
    "求导 f(x)=2x+3x^2+2。",                    # 导数
    "当系数矩阵的行列式不为零时，方程组有唯一解。", # 线代
    "∫₀^(π/2) cos(x) dx = 1。",                 # 积分
    "对 ln(x) 求导。",                          # 导数
]
[[0.4924 0.4658 0.4968 0.4004 0.927  0.3752 0.4866 0.6504]
 [0.601  0.5894 0.4949 0.477  0.542  0.4243 0.557  0.632 ]
 [0.547  0.4868 0.8696 0.445  0.563  0.4531 0.6465 0.552 ]
 [0.502  0.4521 0.4395 0.849  0.4849 0.5645 0.456  0.51  ]
 [0.569  0.4624 0.572  0.4805 0.5503 0.4578 0.7617 0.5264]
 [0.4834 0.4634 0.3574 0.4841 0.4128 0.78   0.391  0.4673]
 [0.8047 0.566  0.5713 0.551  0.5576 0.5103 0.6167 0.5337]
 [0.4954 0.424  0.4312 0.4087 0.735  0.3982 0.4304 0.848 ]]

sentences_1 = ["\text { 求极限 } \lim _{x \rightarrow 0} \frac{\left(1+x^2\right)^3-\left(1-x^2\right)^4}{x^2} \text { ．}"]

sentences_2 = ["\text { 求极限 } \lim _{x \rightarrow+\infty} x\left[\sqrt{x^2+2 x+5}-(x+1)\right] \text { ．}"]
#[[0.8174]]   两道题目都是极限的题目，但是题目不是同一道题
'''

sentences_1 = ["\text { 求极限 } \lim _{x \rightarrow 0} \frac{\left(1+x^2\right)^3-\left(1-x^2\right)^4}{x^2} \text { ．}"]

sentences_2 = ["\text{ 求导数 } \frac{d}{dx}\left[\frac{(1+x^2)^3}{\sqrt{1-x^2}}\right] \text{．}"]
#[[0.8384]]   求导数，且公式差距较大，相似度反而更高


embeddings_1 = model.encode(sentences_1,
                            batch_size=12,
                            max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                            )['dense_vecs']
embeddings_2 = model.encode(sentences_2)['dense_vecs']
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
# [[0.6265, 0.3477], [0.3499, 0.678 ]]
