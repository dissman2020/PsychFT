import logging
import os
import sys

import torch.nn as nn
import wandb
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from transformers.loss.loss_utils import fixed_cross_entropy
from transformers.trainer_utils import get_last_checkpoint
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
    DataCollatorForCompletionOnlyLM,
)

from utils import get_model_name, init_wandb

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
VOCAB_SIZE = 128256  # llama vocab size
accelerator = Accelerator()


class MyCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.9, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index

    def forward(self, logits, labels, step, num_items_in_batch=None):
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        shift_logits = shift_logits.view(-1, VOCAB_SIZE)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = fixed_cross_entropy(shift_logits, shift_labels, num_items_in_batch, self.ignore_index)
        if accelerator.is_main_process:
            wandb.log({"CE Loss": loss.item()}, step=step)
        return loss


class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = MyCrossEntropyLoss(alpha=0.9)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fn(
            logits,
            labels,
            self.state.global_step,
            num_items_in_batch = num_items_in_batch)
        return (loss, outputs) if return_outputs else loss


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    if accelerator.is_main_process:
        init_wandb(get_model_name(training_args.output_dir), "")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(training_args.get_process_log_level())

    last_ckpt = None
    if os.path.isdir(training_args.output_dir):
        last_ckpt = get_last_checkpoint(training_args.output_dir)
    if last_ckpt and training_args.resume_from_checkpoint is None:
        logger.info(f"Resuming from {last_ckpt}")

    ds = load_dataset(script_args.dataset_name)
    train_ds = ds[script_args.dataset_train_split]
    eval_ds = ds[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None

    train_ds = train_ds.map(
        remove_columns=[c for c in train_ds.column_names if c != "text"])  # 手动去掉无用信息
    train_ds = train_ds.map(lambda ex, idx: {**ex, "idx": idx}, with_indices=True)  # 添加索引
    # train_ds = train_ds.shard(num_shards=accelerator.num_processes, index=accelerator.process_index)  # 切分数据

    if eval_ds:
        eval_ds = eval_ds.map(lambda ex, idx: {**ex, "idx": idx}, with_indices=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    tokenizer.pad_token_id = 128004  # llama pad_id
    tokenizer.padding_side = "right"

    collator = DataCollatorForCompletionOnlyLM(response_template=" <<", instruction_template=">>", tokenizer=tokenizer)

    trainer = CustomSFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        dataset_text_field="text",
        peft_config=get_peft_config(model_args),
    )

    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint or last_ckpt)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_ds)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
