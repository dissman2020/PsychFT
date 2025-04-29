import logging
import os
import sys

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    DataCollatorForCompletionOnlyLM,
)

from utils.file_ops import get_model_name
from utils.logs import init_wandb

logger = logging.getLogger(__name__)

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ################
    # Load datasets
    ################
    dataset = load_dataset(script_args.dataset_name)
    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None
    eval_dataset = eval_dataset.filter(lambda example: example['experiment'].startswith("badham2017deficits"))

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    collator = DataCollatorForCompletionOnlyLM(response_template=" <<", instruction_template=">>", tokenizer=tokenizer)

    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = SFTTrainer(
        dataset_text_field=training_args.dataset_text_field,
        model=model_args.model_name_or_path,
        data_collator=collator,
        args=training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args)
    )
    # predictions = trainer.predict(collator(eval_dataset[:2]))
    # predictions = trainer.predict(tokenizer(eval_dataset["text"][:2], return_tensors='pt'))
    # result = trainer.evaluate()
    text_samples = eval_dataset["text"][:16]
    tokenized_inputs = tokenizer(
        text_samples,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    prediction_dataset = datasets.Dataset.from_dict({
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"]
    })

    # 执行预测
    predictions = trainer.predict(prediction_dataset)
    print(predictions)

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
