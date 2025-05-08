import os
import logging
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

LOGITS_DIR = "/data/kankan.lan/repos/psy101/teacher_logits"
VOCAB_SIZE = 128256 # llama vocab size
TOP_K = 10

def init_wandb(model_name, method_name):
    current_time = time.strftime('%Y-%m-%d-%H_%M', time.localtime())
    wandb.init(project="Psych", name=f"{method_name}[{model_name}]({current_time})")

class CrossEntropyKLDLoss(nn.Module):
    def __init__(self, alpha=0.9, ignore_index=-100, kl_reduction='batchmean'):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.kld = nn.KLDivLoss(reduction=kl_reduction)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor,
                      labels: torch.Tensor,
                      teacher_logits: torch.Tensor) -> torch.Tensor:

        B, S, V = logits.size()

        # 1) Shift logits/labels：在 t-1 时刻预测 x_t
        shifted_logits = logits[:, :-1, :].contiguous()   # (B, S-1, V)
        shifted_labels = labels[:, 1:].contiguous()       # (B, S-1)

        # 2) 交叉熵损失：在正确对齐的预测/目标上计算
        flat_logits = shifted_logits.view(-1, V)          # (B*(S-1), V)
        flat_labels = shifted_labels.view(-1)             # (B*(S-1))
        ce_loss = self.ce(flat_logits, flat_labels)

        # 3) KL 散度：只针对那些 labels != ignore_index 的位置
        mask = flat_labels != self.ignore_index           # (B*(S-1))
        if mask.sum() > 0:
            student_log_probs = F.log_softmax(flat_logits[mask], dim=-1)  # (N, V)
            teacher_probs    = F.softmax(teacher_logits, dim=-1)         # (N, V)

            # 若两者行数不一致，则截断到最小行数
            n1 = student_log_probs.size(0)
            n2 = teacher_probs.size(0)
            if n1 != n2:
                n_min = min(n1, n2)
                student_log_probs = student_log_probs[:n_min]
                teacher_probs     = teacher_probs[:n_min]

            kld_loss = self.kld(student_log_probs, teacher_probs)
        else:
            kld_loss = torch.tensor(0.0, device=logits.device)

        total_loss = self.alpha * ce_loss + (1.0 - self.alpha) * kld_loss
        # total_loss = 1 / (self.alpha / ce_loss + (1 - self.alpha) / kld_loss) if kld_loss != 0 else ce_loss

        print(f"[LOSS] CE = {ce_loss.item():.4f}, KL = {kld_loss.item():.4f}, total = {total_loss.item():.4f}")

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

        batch["teacher_logits"] = torch.cat(teacher_list, dim=0)
        return batch

class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = CrossEntropyKLDLoss(alpha=0.9)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        teacher_logits = inputs.pop("teacher_logits")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fn.forward(logits, labels, teacher_logits)

        return (loss, outputs) if return_outputs else loss

def main(script_args, training_args, model_args):
    set_seed(training_args.seed)
    training_args.remove_unused_columns = False  # 阻止Trainer自动剔除idx索引编号
    training_args.max_grad_norm = 1.0
    init_wandb(os.path.basename(model_args.model_name_or_path), "SFT_KL")

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
