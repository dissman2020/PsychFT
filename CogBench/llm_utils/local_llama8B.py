from ..base_classes import LLM
import torch
import sys
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re

class llama8BLLM(LLM):
    def __init__(self, llm_info):
        super().__init__(llm_info)
        # 解析参数：model_path, device, tokenizer_path, model_name (用于识别格式)
        # 如果llm_info有4个元素，则分别代表：模型路径、设备、tokenizer路径、模型名称（如Llama）
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
        if self.model_name and "llama" in self.model_name.lower():
            # Llama模型的对话格式
            self.SYSTEM_PROMPT = None  # Llama通常不使用系统提示
            self.B_INST = "[INST]"
            self.E_INST = "[/INST]"
            self.B_SYS = "<<SYS>>\n"
            self.E_SYS = "\n<<SYS>>\n"
        else:
            # 其他模型默认不添加特殊格式
            self.SYSTEM_PROMPT = None
            self.B_INST = ""
            self.E_INST = ""
            self.B_SYS = ""
            self.E_SYS = ""

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
        if hasattr(self, 'B_INST') and self.B_INST:
            # Llama格式
            if self.SYSTEM_PROMPT:
                return f"{self.B_INST} {self.B_SYS}{self.SYSTEM_PROMPT}{self.E_SYS} {text} {self.E_INST}"
            else:
                return f"{self.B_INST} {text} {self.E_INST}"
        else:
            return text


    def _extract_output(self, generated_text, input_text=None):
        """精准提取 A: ... Option X 或 A: ... X 中的数字"""
        # 匹配 A: ... Option X 或 A: ... X
        match = re.search(
            r'^A:\s*(?:.*?\boption\s+)?(\d+)',  # 允许中间有自然语言
            generated_text,
            re.IGNORECASE | re.MULTILINE
        )
        if match:
            return match.group(1)  # 返回数字部分

        # 如果没有匹配到，回退到原始逻辑
        if not hasattr(self, 'B_INST') or not self.B_INST:
            if input_text and generated_text.startswith(input_text):
                return generated_text[len(input_text):].strip()
            return generated_text.strip()
        else:
            inst_end = f"{self.E_INST}"
            end_idx = generated_text.rfind(inst_end)
            if end_idx == -1:
                return generated_text.strip()
            start_idx = end_idx + len(inst_end)
            return generated_text[start_idx:].strip()


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
                    # "temperature": temp if temp > 0 else None,
                    # "do_sample": True if temp > 0 else False
                }

                # 生成
                with torch.no_grad():
                    outputs = self.model.generate(**generate_kwargs)
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

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