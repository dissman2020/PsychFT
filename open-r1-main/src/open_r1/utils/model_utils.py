from transformers import AutoTokenizer, PreTrainedTokenizer

from trl import ModelConfig


# # 在项目主入口文件（如 main.py）最开头添加：
# import sys
# import os

# # 设置项目根目录（注意是包含src的目录）
# project_root = "/data/kankan.lan/wx/deepseek/open-r1-main/open-r1-main/src"
# sys.path.append(project_root)

# from configs import GRPOConfig, SFTConfig

from ..configs import GRPOConfig, SFTConfig

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def get_tokenizer(
    model_args: ModelConfig, training_args: SFTConfig | GRPOConfig, auto_set_chat_template: bool = True
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template
    elif auto_set_chat_template and tokenizer.get_chat_template() is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    return tokenizer
