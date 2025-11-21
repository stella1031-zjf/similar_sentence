from openai import OpenAI
import pandas as pd
import time
import os
import re

# ------------------- 配置部分 -------------------
API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 或直接写 "sk-xxx"
MODEL_NAME = "qwen-flash"
N_SIMILAR = 3
INPUT_FILE = "D:\working Files\qianwen_input.xlsx"
OUTPUT_FILE = ("D:\working Files\questions_with_similar_chinese.xlsx")
BATCH_SIZE = 10
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def rewrite_question(text, n=3):
    """调用大模型生成 n 条语义相似题目"""
    prompt = (
        f"请为下面这道题生成 {n} 条语义相似但表达不同的题目，保持题意不变。\n"
        f"题目：{text}\n"
        "请严格按照以下要求输出：\n"
        "1. 必须只输出一个 JSON 数组。\n "
        "2. 数组中每个元素必须是字符串。\n"
        "3. 不要输出任何解释、提示、说明文字、换行标题等内容。\n"
        "4. 输出格式必须严格为：[\"句子1\", \"句子2\", \"句子3\"]"

    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你一个专升本题库同义改写模型，擅长生成语义相似但表达不同的高等数学、英语、语文、管理学题目。"},
                {"role": "user", "content": prompt}
        ]
        )
        content = resp.choices[0].message.content
        result = eval(content)   # 将字符串 JSON 转成 python 列表
        return result
    except Exception as e:
        print("API 错误:", e)
        return ["", "", ""]


def clean_text_for_excel(text):
    """清理文本中的Excel不支持字符"""
    if not isinstance(text, str):
        return text
    # 移除或替换可能导致问题的字符
    text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]",'', text)
    return text

def normalize_sims(sims):
    clean = []
    for item in sims:
        if isinstance(item, str):
            clean.append(item.strip())
        else:
            clean.append(str(item))
    return clean


# ------------------- 批量处理 -------------------
target_sheets = ["语文"]
xls = pd.ExcelFile(INPUT_FILE)
with pd.ExcelWriter(OUTPUT_FILE) as writer:
    for sheet in xls.sheet_names:
        if sheet not in target_sheets:
            continue
        df = pd.read_excel(xls, sheet)

        for i in range(len(df)):
            q = df.loc[i, "model_input"]
            if isinstance(q, str) and q.strip():
                print(f"正在处理 {sheet} 第 {i+1} 题")
                sims = rewrite_question(q, N_SIMILAR)
                sims = normalize_sims(sims)
                # 写入 sim1、sim2、sim3
                for k in range(N_SIMILAR):
                    df.loc[i, f"sim{k+1}"] = sims[k] if k < len(sims) else ""

                time.sleep(0.6)  # 限制 QPS，避免 API 限流

                # 每处理 BATCH_SIZE 条题目保存一次
            if (i + 1) % BATCH_SIZE == 0:
                # 修改写入部分的代码：
                try:
                    df.to_excel(writer, sheet_name=sheet, index=False)
                except Exception as e:
                    print(f"写入Excel时出错: {e}")
                    # 可以选择清理数据后重试
                    df_cleaned = df.map(clean_text_for_excel)
                    df_cleaned.to_excel(writer, sheet_name=sheet, index=False)

                print(f"已保存前 {i + 1} 条题目")

        df.to_excel(writer, sheet_name=sheet, index=False)

print("已生成：", OUTPUT_FILE)
