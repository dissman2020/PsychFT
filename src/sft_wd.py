import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import wandb
from accelerate import Accelerator
from datasets import load_dataset
from geomloss import SamplesLoss
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

LOGITS_DIR = "/nas_data/kankan.lan/repos/psy101/teacher_logits"
VOCAB_SIZE = 128256  # llama vocab size
TOP_K = 10

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
accelerator = Accelerator()


class CrossEntropyWassersteinLoss(nn.Module):
    def __init__(self, alpha=0.9, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.wasserstein = SamplesLoss(loss="sinkhorn", p=1, blur=0.05)
        self.ignore_index = ignore_index

    def forward(self, logits, labels, teacher_logits, step, num_items_in_batch=None):
        shift_logits, shift_labels, shift_teacher_logits = self._shift_inputs(logits, labels, teacher_logits)
        ce_loss = self._compute_ce_loss(shift_logits, shift_labels, num_items_in_batch)
        wasserstein_loss = self._compute_wasserstein_loss(shift_logits, shift_labels, shift_teacher_logits)
        total_loss = self._compute_total_loss(ce_loss, wasserstein_loss)
        self._log_losses(ce_loss, wasserstein_loss, total_loss, step)
        return total_loss

    def _shift_inputs(self, logits, labels, teacher_logits):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_teacher_logits = teacher_logits[:, :-1, :].contiguous()  # [B, S-1, V]

        # 确保教师 logits 的序列长度与学生一致
        B, S, V = shift_logits.size()
        B_teacher, S_teacher, V_teacher = shift_teacher_logits.size()
        if S_teacher != S:
            shift_teacher_logits = self._adjust_teacher_logits(shift_teacher_logits, S, S_teacher, V_teacher)

        # Flatten the tokens
        shift_logits = shift_logits.view(-1, VOCAB_SIZE)
        shift_labels = shift_labels.view(-1)
        shift_teacher_logits = shift_teacher_logits.view(-1, shift_teacher_logits.size(-1))
        return shift_logits, shift_labels, shift_teacher_logits

    def _adjust_teacher_logits(self, teacher_logits, student_seq_len, teacher_seq_len, vocab_size):
        if teacher_seq_len < student_seq_len:
            pad_size = student_seq_len - teacher_seq_len
            padding = torch.full((teacher_logits.size(0), pad_size, vocab_size), -1e4, device=teacher_logits.device)
            return torch.cat([teacher_logits, padding], dim=1)
        else:
            return teacher_logits[:, :student_seq_len, :]

    def _compute_ce_loss(self, logits, labels, num_items_in_batch):
        return fixed_cross_entropy(logits, labels, num_items_in_batch, self.ignore_index)

    def _compute_wasserstein_loss(self, logits, labels, teacher_logits):
        # 生成有效标签掩码
        mask = (labels != self.ignore_index)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        # 过滤有效位置
        student_logits_valid = logits[mask]  # (N_valid, V)
        teacher_logits_valid = teacher_logits[mask]  # (N_valid, V)

        # 计算概率分布
        student_probs = torch.softmax(student_logits_valid, dim=-1)
        teacher_probs = torch.softmax(teacher_logits_valid, dim=-1)

        # 计算 Wasserstein 损失
        return self.wasserstein(student_probs, teacher_probs)

    def _compute_total_loss(self, ce_loss, wasserstein_loss):
        return self.alpha * ce_loss + (1 - self.alpha) * wasserstein_loss

    def _log_losses(self, ce_loss, wasserstein_loss, total_loss, step):
        if accelerator.is_main_process:
            wandb.log({
                "Step CE Loss": ce_loss.item(),
                "Step wasserstein Loss": wasserstein_loss.item(),
                "Step total Loss": total_loss.item()
            }, step=step)


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

        # 获取输入序列的统一长度（假设已填充到相同长度）
        seq_length = batch["input_ids"].size(1)

        teacher_list = []
        for idx in idx_list:
            npz_path = os.path.join(LOGITS_DIR, f"{idx:06d}_top{self.top_k}_resp_logits.npz")
            full = restore_full_logits(npz_path, self.vocab_size, device)

            # 截断或填充以匹配输入序列长度
            if full.size(0) < seq_length:
                pad_size = seq_length - full.size(0)
                padding = torch.full((pad_size, self.vocab_size), -1e4, device=device)
                padded_full = torch.cat([full, padding], dim=0)
            else:
                padded_full = full[:seq_length]
            teacher_list.append(padded_full)
        # 确保teacher_logits与模型输出的logits维度对齐
        batch["teacher_logits"] = torch.stack(teacher_list, dim=0)
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
        loss = self.loss_fn(
            logits,
            labels,
            teacher_logits,
            self.state.global_step,
            num_items_in_batch=num_items_in_batch)
        return (loss, outputs) if return_outputs else loss


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    if accelerator.is_main_process:
        init_wandb(get_model_name(training_args.output_dir), "")

    training_args.remove_unused_columns = False  # 阻止Trainer自动剔除idx索引编号

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

    if eval_ds:
        eval_ds = eval_ds.map(lambda ex, idx: {**ex, "idx": idx}, with_indices=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    tokenizer.pad_token_id = 128004  # llama pad_id
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
