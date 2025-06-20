import os
import logging
import sys
import time
import numpy as np
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
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
import swanlab
from utils import get_model_name

LOGITS_DIR = "/data/kankan.lan/repos/psy101/teacher_logits"
VOCAB_SIZE = 128256  # llama vocab size
TOP_K = 10
BETA = 0.8
GAMMA = 0.8
Update_Step = 8

accelerator = Accelerator()

# Token-level decaying weight similarity
def decaying_weight_similarity_tokens(
    a: list,
    b: list,
    beta: float = BETA
) -> float:

    len_a, len_b = len(a), len(b)

    if len_b >= len_a and b[:len_a] == a:
        return 1.0
    max_len = max(len_a, len_b)
    num = 0.0
    den = 0.0
    w = 1.0
    for i in range(max_len):
        den += w
        tok_a = a[i] if i < len_a else None
        tok_b = b[i] if i < len_b else None
        if tok_a is not None and tok_a == tok_b:
            num += w
        w *= beta
    return num / den if den > 0 else 0.0

def init_swanlab(model_name, method_name):
    current_time = time.strftime('%Y-%m-%d-%H_%M', time.localtime())
    swanlab.init(project="Psych", name=f"{method_name}[{model_name}]({current_time})")

class CustomizedLoss(nn.Module):
    def __init__(self, model, ignore_index=-100):
        super().__init__()
        self.model = model
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index
        # 历史loss记录
        self.loss_history = {
            "ce": deque(maxlen=Update_Step),
            "dws": deque(maxlen=Update_Step),
        }

        self.prev_avg_losses = {
            "ce": None,
            "dws": None,
        }

        self.weights = [0.7, 0.3]
        self.step_counter = 0

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: torch.Tensor,
                step: int,
                num_items_in_batch: Optional[int] = None) -> torch.Tensor:

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, S, V)
        B, S, V = logits.size()

        # 1) Shift logits/labels：在 t-1 时刻预测 x_t
        shifted_logits = logits[:, :-1, :].contiguous()  # (B, S-1, V)
        shifted_labels = labels[:, 1:].contiguous()  # (B, S-1)

        # 2) 交叉熵损失：在正确对齐的预测/目标上计算
        flat_logits = shifted_logits.view(-1, V)  # (B*(S-1), V)
        flat_labels = shifted_labels.view(-1)  # (B*(S-1))

        ce_loss = self._compute_ce_loss(flat_logits, flat_labels, num_items_in_batch)

        mask = flat_labels != self.ignore_index  # (B*(S-1))

        # 3) 计算token级相似度
        pred_ids_flat = flat_logits.argmax(dim=-1)

        dws_scores = []
        for i in range(B):
            true_seq = flat_labels[mask].tolist()
            pred_seq = pred_ids_flat[mask].tolist()

            # 计算 dws
            dws_val = decaying_weight_similarity_tokens(true_seq, pred_seq)
            dws_scores.append(dws_val)

        # 批内求平均并取 loss
        dws = torch.tensor(dws_scores, device=logits.device).mean()
        dws_loss = 1.0 - dws

        # 4) 计算总损失
        self.loss_history["ce"].append(ce_loss.detach())
        self.loss_history["dws"].append(dws_loss.detach())
        self.step_counter += 1

        if self.step_counter % Update_Step == 0 and all(len(v) == Update_Step for v in self.loss_history.values()):
            # 计算当前 Update_Step 范围内的平均损失
            current_avg = {
                k: torch.stack(list(v)).mean()
                for k, v in self.loss_history.items()
            }

            for k in ["ce", "dws"]:
                self.prev_avg_losses[k] = current_avg[k]

            # 计算 total_loss_avg 并记录日志
            alpha = self.weights
            total_loss_avg = (alpha[0] * current_avg["ce"] + alpha[1] * current_avg["dws"])

            self._log_losses(current_avg["ce"], current_avg["dws"], total_loss_avg, step, alpha)

        # 使用当前权重计算总损失
        alpha = self.weights
        total_loss = alpha[0] * ce_loss + alpha[1] * dws_loss

        return total_loss

    def _compute_ce_loss(self, logits, labels, num_items_in_batch):
        return fixed_cross_entropy(logits, labels, num_items_in_batch, self.ignore_index)

    def _log_losses(self, ce_loss, dws_loss, total_loss, step, alpha):
        if accelerator.is_main_process:
            print(f"[LOSS] CE = {ce_loss.item():.4f}, DWS = {dws_loss.item():.4f}, total = {total_loss.item():.4f}")
            swanlab.log({
                "Step CE Loss": ce_loss.item(),
                "Step DWS Loss": dws_loss.item(),
                "Step total Loss": total_loss.item()
            }, step=step)


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

        return batch


class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = CustomizedLoss(self.model)

    def compute_loss(self, model, inputs, num_items_in_batch=None):
        labels          = inputs.pop("labels")

        # 计算 num_items_in_batch（有效 token 数）
        with torch.no_grad():
            shifted_labels = labels[:, 1:].contiguous().view(-1)
            mask = shifted_labels != self.loss_fn.ignore_index
            num_items_in_batch = mask.sum().item()

        loss = self.loss_fn.forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
            step=self.state.global_step,
            num_items_in_batch=num_items_in_batch)

        return loss


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)
    training_args.remove_unused_columns = False  # 阻止Trainer自动剔除idx索引编号
    training_args.max_grad_norm = 1.0

    if accelerator.is_main_process:
        init_swanlab(get_model_name(training_args.output_dir), "")

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
