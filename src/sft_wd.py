import os
import logging
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
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
import wandb
from geomloss import SamplesLoss

from utils import get_model_name, init_wandb

LOGITS_DIR = "/data/kankan.lan/repos/psy101/teacher_logits2"
VOCAB_SIZE = 128256 # llama vocab size
TOP_K = 10

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

class CrossEntropyWassersteinLoss(nn.Module):
    def __init__(self, alpha=0.9, ignore_index=-100, kl_reduction='batchmean'):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.kld = nn.KLDivLoss(reduction=kl_reduction)
        self.wasserstein = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)

    def forward(self, logits, labels, teacher_logits):
        B, S, V = logits.size()
        logits_flat = logits.view(-1, V)
        labels_flat = labels.view(-1)

        total_loss = self.ce(logits_flat, labels_flat)
        print(f"[LOSS] Total_loss = {total_loss.item():.4f}")

        # ce_loss = self.ce(logits_flat, labels_flat)
        # student_probs = torch.softmax(logits_flat, dim=-1)
        # teacher_probs = torch.softmax(teacher_logits[: logits_flat.size(0)], dim=-1)
        # # wasserstein_loss = self.wasserstein(student_probs, teacher_probs)
        #
        # total_loss = self.alpha * ce_loss + (1.0 - self.alpha) * wasserstein_loss
        #
        # print(
        #     f"[LOSS] CE = {ce_loss.item():.4f}, Wasserstein = {wasserstein_loss.item():.4f}, Total = {total_loss.item():.4f}")

        return total_loss

# 将保存的 logits 恢复为全量 logits
def restore_full_logits(npz_path, vocab_size, device):
    arr = np.load(npz_path)
    values = torch.tensor(arr["values"], dtype=torch.float32, device=device)
    indices = torch.tensor(arr["indices"], dtype=torch.long, device=device)
    N, K = indices.shape
    full = torch.full((N, vocab_size), fill_value=-1e4, dtype=values.dtype, device=device)
    full.scatter_(1, indices, values)
    return full

class KLDataCollator:
    def __init__(self, tokenizer, top_k, response_template=" <<", instruction_template=">>"):
        self.base_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            instruction_template=instruction_template,
            tokenizer=tokenizer,
        )
        self.tokenizer = tokenizer
        self.vocab_size = VOCAB_SIZE
        self.top_k = top_k

    def __call__(self, features):
        for ex in features:
            if "text" in ex:
                del ex["text"]

        idx_list = [ex["idx"] for ex in features]
        batch = self.base_collator(features)
        labels = batch["labels"]
        device = labels.device

        teacher_list = []
        for idx in idx_list:
            npz_path = os.path.join(LOGITS_DIR, f"{idx:06d}_top{self.top_k}_resp_logits.npz")
            full = restore_full_logits(npz_path, self.vocab_size, device)
            teacher_list.append(full)

        # 确保teacher_logits与模型输出的logits维度对齐
        batch["teacher_logits"] = torch.cat(teacher_list, dim=0).view(-1, self.vocab_size)
        return batch

class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = CrossEntropyWassersteinLoss(alpha=0.9)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        teacher_logits = inputs.pop("teacher_logits")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fn(logits, labels, teacher_logits)
        return (loss, outputs) if return_outputs else loss

def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    init_wandb(get_model_name(model_args.model_name_or_path), "SFT_WD")

    training_args.remove_unused_columns = False  # 阻止Trainer自动剔除idx索引编号
    # init_wandb(os.path.basename(model_args.model_name_or_path), "SFT_KL")

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
        remove_columns=[c for c in train_ds.column_names if c != "text"])                          # 手动去掉无用信息
    train_ds = train_ds.map(lambda ex, idx: {**ex, "idx": idx}, with_indices=True)                 # 添加索引
    train_ds = train_ds.shard(num_shards=training_args.world_size, index=training_args.local_rank) # 切分数据

    if eval_ds:
        eval_ds = eval_ds.map(lambda ex, idx: {**ex, "idx": idx}, with_indices=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    tokenizer.pad_token_id = 128004 #llama pad_id
    tokenizer.padding_side = "right"

    collator = KLDataCollator(tokenizer=tokenizer, top_k=TOP_K)

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

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint or last_ckpt)
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
