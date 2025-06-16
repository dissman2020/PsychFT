import os
import sys
import time
import anthropic
import openai

from .centaur import CENTAURLLM
from .sftklps3meowbetagamma import SFTKLPS3MEOWBETAGAMMALLM
from .sftklps3meowbeta_1 import SFTKLPS3MEOWBETA_1LLM
from .sftklps3meowbeta1 import SFTKLPS3MEOWBETA1LLM
from .sftklps3 import SFTKLPS3LLM
from .sftps3meow import SFTPS3MEOWLLM
from .sftps3 import SFTPS3LLM
from .sftklmeow import SFTKLMEOWLLM
from .cekl import CEKLLLM
from .ce import CELLM
from .local_qwen import QwenLLM
from .local_R1llama8B import R1llama8BLLM
from .local_llama8B import llama8BLLM
from .local_llama1B import llama1BLLM

from ..base_classes import RandomLLM 
from ..base_classes import InteractiveLLM
from dotenv import load_dotenv
# Import scripts for the different LLMs
from .gpt import GPT3LLM, GPT4LLM
from .anthropic import AnthropicLLM
from .google import GoogleLLM
from .hf import HF_API_LLM

def get_llm(engine, temp, max_tokens, with_suffix=False):
    '''
    Based on the engine name, returns the corresponding LLM object
    '''
    # Initialize flags for step back and CoT as False
    step_back = False
    cot = False

    # Check if the engine name is a prompt engineering technique which 
    # therefore requires more tokens and a boolean flag for signal the appending required at the end
    if engine.endswith('_sb'):
        engine = engine[:-3]
        max_tokens = 350
        step_back = True
    elif engine.endswith('_cot'):
        engine = engine[:-4]
        max_tokens = 350
        cot = True

    # Check which engine is being used and assign the corresponding LLM object with the required parameters (e.g: API keys)
    if engine == "interactive":
        llm = InteractiveLLM('interactive')
    elif engine.startswith("text-davinci") or engine.startswith("text-curie") or engine.startswith("text-babbage") or engine.startswith("text-ada"):
        load_dotenv(); gpt_key = os.getenv("OPENAI_API_KEY")
        llm = GPT3LLM((gpt_key, engine, with_suffix))
    elif engine.startswith("gpt"):
        # load_dotenv(); gpt_key = os.getenv(f"OPENAI_API_KEY{2 if engine == 'gpt-4' else ''}")
        load_dotenv(); gpt_key = os.getenv(f"OPENAI_API_KEY")
        llm = GPT4LLM((gpt_key, engine))
    elif engine.startswith("claude"):
        load_dotenv(); anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        llm = AnthropicLLM((anthropic_key, engine))
    elif engine.startswith("hf") or engine.startswith("llama-2") :
        llm = HF_API_LLM((engine, max_tokens, temp))
    elif engine.startswith("local_llama1B"):
        llm = llama1BLLM(("/nas_data/kankan.lan/modelscope_models/LLM-Research/Llama-3.2-1B", "auto", None))
    elif engine.startswith("ce"):
        llm = CELLM(("/nas_data/kankan.lan/repos/psy101/data/Llama-3.2-1B-CE", "auto", None))
    elif engine.startswith("cekl"):
        llm = CEKLLLM(("/nas_data/kankan.lan/repos/psy101/data/Llama-3.2-1B-ce_kl", "auto", None))
    elif engine.startswith("sftklmeow"):
        llm = SFTKLMEOWLLM(("/nas_data/kankan.lan/repos/psy101/data/Llama-3.2-1B-SFT+KL+MEOW", "auto", None))
    elif engine.startswith("sftps3"):
        llm = SFTPS3LLM(("/nas_data/kankan.lan/repos/psy101/data/Llama-3.2-1B-SFT+KL+MEOW", "auto", None))
    elif engine.startswith("sftps3meow"):
        llm = SFTPS3MEOWLLM(("/nas_data/kankan.lan/repos/psy101/data/Llama-3.2-1B-SFT+KL+MEOW", "auto", None))
    elif engine.startswith("sftklps3"):
        llm = SFTKLPS3LLM(("/nas_data/kankan.lan/repos/psy101/data/Llama-3.2-1B-SFT+KL+PS3", "auto", None))
    elif engine.startswith("sftklps3meowbeta1"):
        llm = SFTKLPS3MEOWBETA1LLM(("/nas_data/kankan.lan/repos/psy101/data/Llama-3.2-1B-Adaptive-Weight", "auto", None))
    elif engine.startswith("sftklps3meowbeta-1"):
        llm = SFTKLPS3MEOWBETA_1LLM(("/nas_data/kankan.lan/repos/psy101/data/Llama-3.2-1B-Adaptive-Weight-test", "auto", None))
    elif engine.startswith("sftklps3meowbetagamma"):
        llm = SFTKLPS3MEOWBETAGAMMALLM(("/nas_data/kankan.lan/repos/psy101/data/Llama-3.2-1B-Adaptive-Weight-With-Gamma", "auto", None))
    elif engine.startswith("local_llama8B"):
        llm = llama8BLLM(("/nas_data/kankan.lan/modelscope_models/LLM-Research/Meta-Llama-3.1-8B", "auto", None))
    elif engine.startswith("local_R1llama8B"):
        llm = R1llama8BLLM(("/data/kankan.lan/modelscope_models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "auto", None))
    elif engine.startswith("centaur"):
        llm = CENTAURLLM(("/data/kankan.lan/repos/Llama-3.1-Centaur-8B-adapter/Centaur-8B", "auto", None))
    elif engine.startswith("local_qwen"):
        llm = QwenLLM(("/data/kankan.lan/wx/model/base/Qwen2.5-1.5B-Instruct", "auto", None))
    # elif engine.startswith("gemini"):
    #     load_dotenv(); gemini_key = os.getenv("GOOGLE_CREDENTIALS_FILENAME2")
    #     llm = GeminiLLM((gemini_key, engine))
    #     llm.is_gemini = True #See the TODO below
    elif ('bison' in engine):
        load_dotenv(); google_key = os.getenv("GOOGLE_CREDENTIALS_FILENAME2")
        llm = GoogleLLM((google_key, engine))
    else:
        print('No key found')
        llm = RandomLLM(engine)

    #TODO: I am thinking for some models which really are stubborn/tedious processing, to maybe set some flag here in the form is_X = True which could eb recognized in the experiments to mitigate the issues? For now no, to keep things simple and not hardcoded for each LLM.
    # if not hasattr(llm, 'is_X'):
    #     llm.is_X = False
        
    # Set temperature and max_tokens
    llm.temperature = temp
    llm.max_tokens = max_tokens
    llm.step_back = step_back
    llm.cot = cot
    return llm

       