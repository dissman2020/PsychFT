from trl import DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer

model_path = "/data/kankan.lan/llms/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path)

input_text = ["You encounter a new problem with a new rule determining which objects belong to each category:\
    You see a big white square. You press <<C>>. The correct category is N.\
    You see a big black triangle. You press <<C>>. The correct category is C.\
    You see a small white square. You press <<N>>. The correct category is N.",
    "Block 0, N = 2:\
    You see the letter v.\
    You see the letter d and press <<T>>.\
    You see the letter g and press <<T>>."]

collator = DataCollatorForCompletionOnlyLM(response_template=" <<", instruction_template=">>", tokenizer=tokenizer)
print(collator(input_text))
