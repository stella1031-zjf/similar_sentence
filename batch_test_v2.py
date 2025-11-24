from FlagEmbedding import BGEM3FlagModel
import torch
import pandas as pd


def batched_compute_score(model, pairs, batch_size=32):
    results = []
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i + batch_size]
        batch_scores = model.compute_score(
            batch_pairs,
            max_passage_length=128,
            weights_for_different_modes=[0.5, 0, 0.5]
        )
        results.extend(batch_scores["colbert+sparse+dense"])
    return results


def main():
    # Check if CUDA is available
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    model_path = "/home/zjf/python_projects/similaritysentence/models/bge-m3"

    model = BGEM3FlagModel(
        model_path,
        use_fp16=True,
        device='cuda',
        use_safetensors=True
    )

    INPUT_FILE = r"/home/zjf/python_projects/similaritysentence/test_files/questions_with_similar_math.xlsx"
    OUTPUT_FILE = r"/home/zjf/python_projects/similaritysentence/test_files/similarity_math.xlsx"
    xls = pd.ExcelFile(INPUT_FILE)

    processed_sheets = {}
    for sheet in xls.sheet_names:
        print(f"\n开始处理 Sheet：{sheet}")

        df = pd.read_excel(xls, sheet)

        if not {"model_input", "sim1"}.issubset(df.columns):
            print(f"Sheet {sheet} 缺少 model_input 或 sim1，跳过")
            continue

        # Read two columns
        original_list = df["model_input"].astype(str).tolist()
        sim1_list = df["sim1"].astype(str).tolist()

        # Build pairs = [[sent1, sent2], ...]
        pairs = [[o, s] for o, s in zip(original_list, sim1_list)]

        print(f"正在计算相似度，共 {len(pairs)} 条句子对...")

        # Get colbert+sparse+dense mode
        sim_values = batched_compute_score(model, pairs, batch_size=32)

        # Convert to regular float (avoid Excel write errors)
        df["similarity1and2"] = [float(x) for x in sim_values]
        # Store processed DataFrame
        processed_sheets[sheet] = df
        print(f"✔ Sheet {sheet} 处理完成，已写入 similarity1and2")

    # Write all sheets at once
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        for sheet_name, dataframe in processed_sheets.items():
            dataframe.to_excel(writer, sheet_name=sheet_name, index=False)

    print("\n🎉 所有 Sheet 已处理完毕！输出文件：", OUTPUT_FILE)


if __name__ == '__main__':
    main()
