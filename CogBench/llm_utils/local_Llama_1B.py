# from ..base_classes import LLM
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# class LocalLLM(LLM):
#     def __init__(self, llm_info):
#         """
#         :param llm_info: 元组格式 (model_path, device, tokenizer_path)
#                          示例: ("/path/to/model", "auto", None)
#         """
#         super().__init__(llm_info)
#         model_path, device, tokenizer_path = llm_info
        
#         # 硬件自动检测
#         self.device = device.lower()
#         if self.device == "auto":
#             self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
#         # 模型配置（可在此处硬编码你的本地模型路径）
#         self.model_path = model_path or "/data/kankan.lan/wx/model/base/Qwen2.5-1.5B-Instruct"  # 默认路径
#         self.tokenizer_path = tokenizer_path or self.model_path
        
#         # 加载模型
#         self._load_model()

#     def _load_model(self):
#         """加载本地模型"""
#         print(f"Loading local model from: {self.model_path}")
#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 self.tokenizer_path,
#                 trust_remote_code=True
#             )
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 self.model_path,
#                 device_map="auto" if self.device == "cuda" else None,
#                 torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
#                 trust_remote_code=True
#             )
#         except Exception as e:
#             raise RuntimeError(f"Failed to load local model: {str(e)}")

#     def _generate(self, text, temp, max_tokens):
#         """生成响应"""
#         print(f"Generating......")
#         inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
#         outputs = self.model.generate(
#             **inputs,
#             max_new_tokens=max_tokens,
#             temperature=temp,
#             do_sample=True if temp > 0 else False
#         )
#         return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

from ..base_classes import LLM
import torch
import sys
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class LocalLLM(LLM):
    def __init__(self, llm_info):
        super().__init__(llm_info)
        # 解析参数：model_path, device, tokenizer_path, model_name (用于识别格式)
        # 如果llm_info有4个元素，则分别代表：模型路径、设备、tokenizer路径、模型名称（如Qwen）
        if len(llm_info) == 4:
            model_path, device, tokenizer_path, model_name = llm_info
        else:
            model_path, device, tokenizer_path = llm_info
            model_name = None

        # 设备配置
        self.device = device.lower()
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.model_name = model_name  # 用于识别是否需要特殊对话格式

        # 根据模型名称设置对话格式
        if self.model_name and "qwen" in self.model_name.lower():
            self.SYSTEM_PROMPT = "You are a helpful assistant."
            self.IM_START = "<|im_start|>"
            self.IM_END = "<|im_end|>"
        else:
            # 其他模型默认不添加特殊格式，但可以根据需要添加
            self.SYSTEM_PROMPT = None
            self.IM_START = ""
            self.IM_END = ""

        # 加载模型
        self._load_model()

    def _load_model(self):
        """加载模型和tokenizer"""
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=True
            )

            # 配置模型加载参数
            torch_dtype = torch.float16 if "cuda" in self.device else torch.float32
            device_map = "auto" if (self.device == "cuda" and torch.cuda.device_count() > 1) else self.device

            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True
            )

            # 如果设备不是cuda（即单设备）且不是多GPU，则需要手动移动模型
            if device_map is None or isinstance(device_map, str):
                self.model.to(self.device)

            # 设置为评估模式
            self.model.eval()

            print(f"Model loaded on device: {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load the model: {str(e)}")

    def _build_input(self, text):
        """根据模型类型构建输入"""
        if self.SYSTEM_PROMPT is not None:
            # Qwen格式
            return f"{self.IM_START}system\n{self.SYSTEM_PROMPT}{self.IM_END}\n" \
                   f"{self.IM_START}user\n{text}{self.IM_END}\n" \
                   f"{self.IM_START}assistant\n"
        else:
            return text

    def _extract_output(self, generated_text, input_text=None):
        """从生成的文本中提取回复部分（只取新增部分）"""
        # 如果没有设置特殊格式，则直接返回整个生成（但去掉输入部分）
        if self.SYSTEM_PROMPT is None:
            # 注意：生成的文本可能包含了输入。我们只想要新生成的部分。
            # 如果生成文本以输入文本开头，则去掉输入文本部分
            if input_text and generated_text.startswith(input_text):
                return generated_text[len(input_text):].strip()
            else:
                # 如果模型名称是qwen，但之前没有设置格式，可能是因为没有指定模型名称，但实际上是qwen
                # 尝试查找结束符
                if '<|im_end|>' in generated_text:
                    start_marker = '<|im_start|>assistant\n'
                    start_idx = generated_text.rfind(start_marker)
                    if start_idx != -1:
                        start_idx += len(start_marker)
                        end_idx = generated_text.find('<|im_end|>', start_idx)
                        if end_idx != -1:
                            return generated_text[start_idx:end_idx].strip()
                # 否则返回整个文本（这种情况可能是模型在继续输入文本）
                return generated_text.strip()
        else:
            # Qwen格式：回复部分在assistant开始标记之后，结束标记之前
            # 注意：生成的内容可能包含结束标记
            assistant_start = f"{self.IM_START}assistant\n"
            # 查找最后一个assistant_start（因为生成的内容中可能包含重复的标记）
            start_idx = generated_text.rfind(assistant_start)
            if start_idx == -1:
                # 如果没有找到，返回整个文本
                return generated_text.strip()
            start_idx += len(assistant_start)
            # 从start_idx开始找结束标记
            end_idx = generated_text.find(self.IM_END, start_idx)
            if end_idx == -1:
                return generated_text[start_idx:].strip()
            else:
                return generated_text[start_idx:end_idx].strip()

    def _generate(self, text, temp, max_tokens):
        """生成文本，带有重试机制"""
        original_text = text  # 保存原始输入，用于后续提取
        text = self._build_input(text)  # 构建模型输入

        # 重试机制
        for i in range(10):
            try:
                # 准备输入
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    return_attention_mask=True
                ).to(self.device)

                # 生成参数
                generate_kwargs = {
                    "input_ids": inputs.input_ids,
                    "attention_mask": inputs.attention_mask,
                    "max_new_tokens": max_tokens,
                    "pad_token_id": self.tokenizer.eos_token_id or self.tokenizer.pad_token_id,
                    "temperature": temp if temp > 0 else None,
                    "do_sample": True if temp > 0 else False
                }
                # 移除None值
                generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

                # 生成
                with torch.no_grad():
                    outputs = self.model.generate(**generate_kwargs)
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

                # 提取回复部分
                result = self._extract_output(generated_text, original_text)
                return result

            except Exception as e:
                print(f"Error in generation (attempt {i+1}): {str(e)}")
                time.sleep(3 ** i)  # 指数退避
                # 如果是CUDA内存不足，尝试清空缓存
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
        # 重试10次后失败
        raise RuntimeError("Failed to generate text after 10 attempts.")