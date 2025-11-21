import pandas as pd

INPUT_FILE = "D:\working Files\题库_文本格式.xlsx"
OUTPUT_FILE = "D:\working Files\qianwen_input.xlsx"

xls = pd.ExcelFile(INPUT_FILE)

# 使用 with 语句自动处理保存
with pd.ExcelWriter(OUTPUT_FILE) as writer:
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet)

        # 只保留 question_full
        out_df = df[["model_input"]].copy()

        out_df.to_excel(writer, sheet_name=sheet, index=False)

print("已生成:", OUTPUT_FILE)
