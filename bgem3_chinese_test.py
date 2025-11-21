from FlagEmbedding import BGEM3FlagModel
import torch
import pandas as pd

#检查是否有显卡可用
print(torch.cuda.is_available())
#可用的显卡有几张
print(torch.cuda.device_count ())

model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True, device='cuda',use_safetensors=True)

INPUT_FILE = r"D:\working Files\questions\questions_with_similar_chinese.xlsx"
OUTPUT_FILE = r"D:\working Files\questions\similarity_chinese.xlsx"
xls = pd.ExcelFile(INPUT_FILE)


processed_sheets = {}
for sheet in xls.sheet_names:
    print(f"\n开始处理 Sheet：{sheet}")

    df = pd.read_excel(xls, sheet)


    if not {"model_input", "sim1"}.issubset(df.columns):
        print(f"Sheet {sheet} 缺少 model_input 或 sim1，跳过")
        df.to_excel(writer, sheet_name=sheet, index=False)
        continue

        # 读取两列
    original_list = df["model_input"].astype(str).tolist()
    sim1_list = df["sim1"].astype(str).tolist()

        # 构建 pairs = [[sent1, sent2], ...]
    pairs = [[o, s] for o, s in zip(original_list, sim1_list)]

    print(f"正在计算相似度，共 {len(pairs)} 条句子对...")

    scores = model.compute_score(
            pairs,
            max_passage_length=128,
            weights_for_different_modes=[0.5, 0, 0.5]
        )

        # 取 colbert+sparse+dense 模式
    sim_values = scores["colbert+sparse+dense"]

        # 转为普通 float（避免 Excel 写入错误）
    df["similarity1and2"] = [float(x) for x in sim_values]
    # 存储处理后的 DataFrame
    processed_sheets[sheet] = df
    print(f"✔ Sheet {sheet} 处理完成，已写入 similarity1and2")

    # 统一写入所有 sheet
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        for sheet_name, dataframe in processed_sheets.items():
            dataframe.to_excel(writer, sheet_name=sheet_name, index=False)

    print("\n🎉 所有 Sheet 已处理完毕！输出文件：", OUTPUT_FILE)
