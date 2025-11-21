import pandas as pd
import re
import json
# -------- 1. 去 HTML 标签的函数 --------
def clean_html(raw_html):
    if pd.isna(raw_html):
        return ""
    # 去掉 HTML 标签
    cleanr = re.compile('<.*?>')
    text = re.sub(cleanr, '', str(raw_html))

    # 去掉 &nbsp; 等
    text = text.replace("&nbsp;", " ").strip()
    return text

#-------- 2. 去 json 格式的函数 --------
def clean_json(json_str):

    if pd.isna(json_str):
        return ""

    try:
        options = json.loads(json_str)
        result = []
        for item in options:
            key = item.get("key", "")
            value = item.get("value", "")
            result.append(f"{key} {value}")

        return " ".join(result)

    except Exception:
        # 如果不是 JSON 格式，就原样返回
        return str(json_str)


#-------- 3. 去 latex 格式的函数 --------
def clean_math_format(text):
    """
    去掉数学题目中的 LaTeX 公式格式，保留文本内容和公式核心。
    1. 去掉 $$ 包裹的公式标记
    2. 去掉 \left, \right, \text{}
    3. 去掉多余空格
    """
    if pd.isna(text):
        return ""

    # 去掉 \text{}，保留内部文本
    text = re.sub(r"\\text\{(.*?)\}", r"\1", str(text))

    # 去掉 \left 和 \right
    text = text.replace(r"\left", "").replace(r"\right", "")

    text = re.sub(r"\\begin\{array\}\{.*?\}", "", text)
    text = text.replace(r"\end{array}", "")
    text = re.sub(r"\\frac\{(.*?)\}\{(.*?)\}", r"(\1)/(\2)", text)

    # 去掉 $$ 包裹符号
    text = text.replace("$$", "")

    # 可选：去掉 \( \) 包裹符号
    text = text.replace(r"\(", "").replace(r"\)", "")

    # 去掉多余空格
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# -------- 4. 加载 Excel --------
file_path = "D:\working Files\查重优化题库提供数据.xlsx"
xls = pd.ExcelFile(file_path)

cleaned_sheets = {}

for sheet_name in xls.sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # 只处理 C 和 D 列（你的题干和选项）
    df["题干文本"] = df["题干"].apply(clean_html).apply(clean_math_format)
    df["选项文本"] = df["选项"].apply(clean_html).apply(clean_json).apply(clean_math_format)

    # 合并为模型输入格式（你可自定义）
    df["model_input"] = df["题干文本"] + " " + df["选项文本"]
    cleaned_sheets[sheet_name] = df

# -------- 5. 保存为 new.xlsx --------
with pd.ExcelWriter(r"D:\working Files\题库_文本格式.xlsx") as writer:
    for sheet_name, df in cleaned_sheets.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print("处理完成，输出为：题库_cleaned.xlsx")
