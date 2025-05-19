import jsonlines
from vllm import LLM, SamplingParams

# MODEL_PATH = "/data/kankan.lan/modelscope_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_PATH = "/data/kankan.lan/modelscope_models/deepseek-ai/DeepSeek-R1-Distill-Llama-8"
# INPUT_FILE = "dataset/data_example.jsonl"
INPUT_FILE = "/data/kankan.lan/datasets/marcelbinz/Psych-101-merged/test.jsonl"
OUTPUT_FILE = "dataset/compressed_test.jsonl"
BATCH_SIZE = 4  # 根据GPU显存调整
MAX_INPUT_LENGTH = 32768  # 输入文本最大长度（32KB）
MAX_OUTPUT_LENGTH = 8192  # 输出文本最大长度（8KB）


def generate_compressed_text(text):
    llm = LLM(model=MODEL_PATH)

    prompt = f"""
            Please compress the following text to 75% of its current length while preserving all key information and semantics. Remove redundant content. Output ONLY the compressed text without any additional text.
            Original text:
            {text}

            Compressed text:
            """
    # prompts = [
    #     "Hello, my name is",
    #     "The president of the United States is",
    #     "The capital of France is",
    #     "The future of AI is",
    # ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    outputs = llm.generate(prompt, sampling_params)
    generated_text = outputs[0].outputs[0].text
    return generated_text


def process_jsonl():
    with jsonlines.open(INPUT_FILE) as reader, jsonlines.open(OUTPUT_FILE, mode='w') as writer:
        for item in reader:
            original_text = item.get("text", "")  # 假设字段名为"text"
            if len(original_text) > MAX_OUTPUT_LENGTH:
                compressed = generate_compressed_text(original_text)
                item["compressed_text"] = compressed  # 输出字段为英文
            writer.write(item)


if __name__ == "__main__":
    process_jsonl()
