import os
import numpy as np
import torch
from transformers import AutoTokenizer

# 配置参数
LOGITS_DIR = "/data/kankan.lan/repos/psy101/teacher_logits3"  # logits存储路径
VOCAB_SIZE = 128256  # 与原始模型一致
MODEL_NAME = "/data/kankan.lan/modelscope_models/LLM-Research/Llama-3.2-1B"
TOP_K = 10


def restore_full_logits(npz_path, vocab_size, device="cpu"):
    arr = np.load(npz_path)
    values = torch.tensor(arr["values"], dtype=torch.float32, device=device)
    indices = torch.tensor(arr["indices"], dtype=torch.long, device=device)
    N, K = indices.shape
    full = torch.full((N, vocab_size), fill_value=-1e4, dtype=values.dtype, device=device)
    full.scatter_(1, indices, values)
    return full


def logits_to_text(logits, tokenizer):
    """将logits转换为文本"""
    token_ids = torch.argmax(logits, dim=-1)  # 取每个位置概率最大的token
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return text


def main():
    # 初始化tokenizer（需与生成logits的模型一致）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token_id = 128004  # 与原代码对齐

    # 遍历logits目录
    npz_files = sorted(
        [f for f in os.listdir(LOGITS_DIR) if f.endswith(".npz")],
        key=lambda x: int(x.split("_")[0])  # 按idx排序
    )

    for filename in npz_files:
        idx = filename.split("_")[0]
        npz_path = os.path.join(LOGITS_DIR, filename)

        # 恢复logits
        logits = restore_full_logits(npz_path, VOCAB_SIZE, device="cpu")

        # 转换为文本
        text = logits_to_text(logits, tokenizer)

        # 打印结果
        print(f"===== 样本 {idx} 生成的文本 =====")
        print(text + "\n")


if __name__ == "__main__":
    main()