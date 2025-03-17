from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
import pandas as pd
import argparse
import torch
import json
import os

from utils.logs import init_wandb

# 减少显存碎片化问题
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def get_model_name(name_or_path):
    return os.path.basename(name_or_path.rstrip("/"))


def get_file_path(dir, name_or_path):
    import time
    current_time = time.strftime('_%Y-%m-%d_%H%M', time.localtime(time.time()))
    file_name = get_model_name(name_or_path) + current_time + '.csv'
    return os.path.join(dir, file_name)


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Evaluate a causal language model on various tasks.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    # 从 JSON 文件中加载配置参数
    with open(args.config, 'r') as f:
        config = json.load(f)

    model_name_or_path = config.get("model_name_or_path")
    eval_dataset_name = config.get("eval_dataset")
    max_seq_length = config.get("max_seq_length", 32768)  # 默认值为 32768
    output_dir = config.get("output_dir", "results")
    run_tag = config.get("run_tag", "")
    if not model_name_or_path or not eval_dataset_name:
        raise ValueError("Model and eval_dataset must be specified in the config file.")

    # 初始化 wandb
    init_wandb(get_model_name(model_name_or_path), run_tag)

    task_names = [
        "badham2017deficits",
        "bahrami2020four",
        "enkavi2019adaptivenback",
        "enkavi2019digitspan",
        "enkavi2019gonogo",
        "enkavi2019recentprobes",
        "feng2021dynamics",
        "flesch2018comparing",
        "frey2017cct",
        "frey2017risk",
        "gershman2018deconstructing",
        "gershman2020reward",
        "hebart2023things",
        "hilbig2014generalized",
        "kool2016when",
        "kool2017cost",
        "lefebvre2017behavioural",
        "levering2020revisiting",
        "ludwig2023human",
        "peterson2021using",
        "plonsky2018when",
        "ruggeri2022globalizability",
        "sadeghiyeh2020temporal",
        "schulz2020finding",
        "somerville2017charting",
        "speekenbrink2008learning",
        "steingroever2015data",
        "tomov2020discovery",
        "tomov2021multitask",
        "waltz2020differential",
        "wilson2014humans",
        "wu2023chunking",
        "wulff2018description",
        "wulff2018sampling",
        "xiong2023neural",
        "zorowitz2023data",
    ]

    # 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    l_id = tokenizer(" <<").input_ids[1:]
    r_id = tokenizer(">>").input_ids[1:]
    collator = DataCollatorForCompletionOnlyLM(response_template=l_id, instruction_template=r_id, tokenizer=tokenizer)
    # 加载数据集
    dataset = load_dataset(eval_dataset_name)

    data = []
    total_steps = len(task_names)
    steps = 0
    with torch.no_grad():
        for task_name in task_names:
            steps += 1
            eval_dataset = dataset['test'].filter(lambda example: example['experiment'].startswith(task_name))

            training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_eval_batch_size=1
            )
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=eval_dataset,
                eval_dataset=eval_dataset,
                dataset_text_field="text",
                max_seq_length=max_seq_length,
                data_collator=collator,
            )
            result = trainer.evaluate()

            print(f'{steps}/{total_steps}: {task_name}', flush=True)
            print(result, flush=True)
            data.append([task_name, result['eval_loss']])

        # 保存结果
        df = pd.DataFrame(data, columns=['task', model_name_or_path])
        print(df, flush=True)

        df.to_csv(get_file_path(output_dir, model_name_or_path))
