import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd


def test_model_on_dataset(
        model_name: str,
        dataset_path: str,
        output_path: str,
        text_column: str = "text",
        device: str = None,
        max_length: int = 8192,
        num_return_sequences: int = 1,
):

    # 选择设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    # 读取数据集
    df = pd.read_json(dataset_path, lines=True)
    df = df.head(1)

    # 打开输出文件
    with open(output_path, "w", encoding="utf-8") as writer:
        for idx, row in df.iterrows():
            input_text = row[text_column]
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)

            with torch.no_grad():
                outputs1 = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    do_sample=False,
                )
                outputs2 = model(**inputs,
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    do_sample=False)
            pred_text1 = tokenizer.decode(outputs1.view(-1), skip_special_tokens=True)
            pred_text2 = tokenizer.decode(torch.argmax(outputs2.logits, dim=-1).view(-1), skip_special_tokens=True)

            # 写入文件
            for i, output_ids in enumerate(outputs1):
                pred_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                writer.write(f"==== Sample {idx}, Prediction #{i + 1} ====\n")
                writer.write(f"原文：{input_text}\n")
                writer.write(f"预测：{pred_text}\n\n")

    print(f"已将结果保存到 {output_path}")


if __name__ == "__main__":
    test_model_on_dataset(
        model_name="/data/kankan.lan/modelscope_models/marcelbinz/Llama-3.1-Centaur-8B",
        dataset_path="/data/kankan.lan/datasets/marcelbinz/Psych-101-all/train.jsonl",
        output_path="predictions.txt",
        text_column="text",
        max_length=8192,
        num_return_sequences=1,
    )
