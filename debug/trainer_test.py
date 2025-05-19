import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DataCollatorForCompletionOnlyLM

checkpoint = "/data/kankan.lan/modelscope_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)


input_text = ["You see a big white square. You press <<C>>. The correct category is N.\
    You see a big black triangle. You press <<C>>. The correct category is C.\
    You see a small white square. You press <<N>>. The correct category is N.",
    "You see the letter v.\
    You see the letter d and press <<T>>.\
    You see the letter g and press <<T>>."]

tokens = tokenizer(input_text,
                 truncation=True,
                 padding=True,
                 return_tensors="pt")
collator = DataCollatorForCompletionOnlyLM(response_template=" <<", instruction_template=">>", tokenizer=tokenizer)
# print(tokens)
c_ids = collator(tokens.input_ids)

# 生成模型输入（包含掩码标签）
model_inputs = collator([{"input_ids": ids} for ids in tokens.input_ids])

# 计算loss
with torch.no_grad():
    outputs = model(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        labels=model_inputs["labels"],
    )

loss = outputs.loss
print(f"Total loss: {loss.item():.4f}")

# 计算每个样本的loss
losses = []
for i in range(len(input_text)):
    sample_inputs = {
        "input_ids": model_inputs["input_ids"][i:i + 1],
        "attention_mask": model_inputs["attention_mask"][i:i + 1],
        "labels": model_inputs["labels"][i:i + 1]
    }

    with torch.no_grad():
        outputs = model(**sample_inputs)
        print(f"Text: {input_text[:50]}...\n")
        print(outputs)

    losses.append(outputs.loss.item())

for i, (text, loss) in enumerate(zip(input_text, losses)):
    print(f"Sample {i + 1} loss: {loss:.4f}")
    print(f"Text: {text[:50]}...\n")



