import os

import wandb

from utils.logs import init_wandb

if __name__ == '__main__':
    model_name = "/data/kankan.lan/modelscope_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    init_wandb(os.path.basename(model_name.rstrip("/")), "SFT")
    for epoch in range(5):
        wandb.log({"epoch": epoch, "train_loss": epoch})
