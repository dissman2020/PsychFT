import os
import re
import glob
import json
import time
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import Accelerator
from tqdm.auto import tqdm

# regex to find all <<...>> segments
SEGMENT_PATTERN = re.compile(r"(<<.*?>>)", re.DOTALL)

class TextDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch, tokenizer, max_seq_length):
    texts = [item["text"] for item in batch]
    idxs = [item["idx"] for item in batch]

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_offsets_mapping=True,
    )
    offsets = enc.pop("offset_mapping")
    batch_size, seq_len = enc["input_ids"].shape
    masks = torch.zeros((batch_size, seq_len), dtype=torch.bool)

    for i, text in enumerate(texts):
        spans = [m.span(1) for m in SEGMENT_PATTERN.finditer(text)]
        for tok_idx, (start_char, end_char) in enumerate(offsets[i].tolist()):
            for seg_start, seg_end in spans:
                if start_char >= seg_start and end_char <= seg_end:
                    masks[i, tok_idx] = True
                    break

    enc["response_mask"] = masks
    enc["idxs"] = torch.tensor(idxs, dtype=torch.long)
    return enc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, default="/data/kankan.lan/modelscope_models/marcelbinz/Llama-3.1-Centaur-8B")
    parser.add_argument("--dataset_name", type=str, default="/data/kankan.lan/datasets/marcelbinz/Psych-101-all")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_sequence_length", type=int, default=1024)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="/data/kankan.lan/repos/psy101/teacher_logits_Centaur-8B")
    args = parser.parse_args()

    accelerator = Accelerator()
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    # Detect already processed samples (use JSON + sync)
    processed_file = os.path.join(args.output_dir, f"processed_top{args.top_k}.json")
    if accelerator.is_main_process:
        pattern = os.path.join(args.output_dir, f"*_top{args.top_k}_resp_logits.npz")
        existing_files = glob.glob(pattern)
        processed_idxs = set()
        for f in existing_files:
            basename = os.path.basename(f)
            idx_str = basename.split('_')[0]
            if idx_str.isdigit():
                processed_idxs.add(int(idx_str))
        with open(processed_file, "w") as f:
            json.dump(sorted(processed_idxs), f)
        print(f"Saved processed idxs to {processed_file} ({len(processed_idxs)} entries)")

    accelerator.wait_for_everyone()

    while not os.path.exists(processed_file):
        time.sleep(1)
    with open(processed_file, "r") as f:
        processed_idxs = set(json.load(f))

    raw_ds = load_dataset(args.dataset_name, split=args.dataset_split)
    texts = raw_ds[args.text_field]
    items = [
        {"text": text, "idx": idx}
        for idx, text in enumerate(texts)
        if idx not in processed_idxs
    ]

    ds = TextDataset(items)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, use_fast=True)
    tokenizer.pad_token_id = 128004 #llama pad_id
    tokenizer.padding_side = "right"
    def _collate(batch):
        return collate_fn(batch, tokenizer, args.max_sequence_length)

    dataloader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate,
        drop_last=False,
    )

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        # quantization_config=bnb_config,
    )
    model.eval()

    model, dataloader = accelerator.prepare(model, dataloader)

    for batch in tqdm(dataloader, desc="Inference", disable=not accelerator.is_main_process):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        response_mask = batch["response_mask"]
        idxs = batch["idxs"].tolist()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        for i, sample_idx in enumerate(idxs):
            # mask = response_mask[i]
            # resp_logits = logits[i][mask]
            # if resp_logits.numel() == 0:
            #     continue
            values, indices = torch.topk(logits[i], k=args.top_k, dim=-1)
            values_np = values.to(torch.float32).cpu().numpy()
            indices_np = indices.to(torch.float32).cpu().numpy()
            out_path = os.path.join(
                args.output_dir,
                f"{sample_idx:06d}_top{args.top_k}_resp_logits.npz"
            )
            if not os.path.exists(out_path):
                np.savez(out_path, values=values_np, indices=indices_np)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"All top-{args.top_k} logits saved to {args.output_dir}")


if __name__ == "__main__":
    main()