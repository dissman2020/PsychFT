def init_wandb(model_name, method_name):
    import wandb
    import time
    current_time = time.strftime('%Y-%m-%d-%H_%M', time.localtime(time.time()))
    wandb.init(project="Psych", name=f"{method_name}[{model_name}]({current_time})")

if __name__ == "__main__":
    init_wandb("test_model", "test_method")