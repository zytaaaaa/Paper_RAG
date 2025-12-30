import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.language_models import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Optional, Tuple

# 定义模型路径（请根据实际下载路径修改）
model_path = './IEITYuan/Yuan2-2B-Mars-hf'
embedding_model_path = './AI-ModelScope/bge-small-en-v1___5'

# 定义模型数据类型
torch_dtype = torch.bfloat16  # A10


class Yuan2_LLM(LLM):
    """
    Wrapper for Yuan2 model (moved from app.py)
    """
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, mode_path: str = model_path):
        super().__init__()
        print("Create tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(mode_path, add_eos_token=False, add_bos_token=False,
                                                       eos_token='<eod>', pad_token='<pad>', trust_remote_code=True)
        self.tokenizer.add_tokens(
            ['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>', '<commit_before>',
             '<commit_msg>', '<commit_after>', '<jupyter_start>', '<jupyter_text>', '<jupyter_code>',
             '<jupyter_output>', '<empty_output>'], special_tokens=True)

        print("Create model...")
        self.model = AutoModelForCausalLM.from_pretrained(mode_path, torch_dtype=torch.bfloat16,
                                                          trust_remote_code=True).cuda()

    def _call(
            self,
            prompt: str,
            stop: Optional[list] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs,
    ) -> str:
        prompt = prompt.strip()
        prompt += "<sep>"
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
        outputs = self.model.generate(
            inputs,
            do_sample=False,
            max_length=4096,
            use_cache=False,  # 关键修复：禁用cache避免past_key_values错误
            pad_token_id=self.tokenizer.eos_token_id
        )
        output = self.tokenizer.decode(outputs[0])
        response = output.split("<sep>")[-1].split("<eod>")[0]
        return response

    @property
    def _llm_type(self) -> str:
        return "Yuan2_LLM"


@st.cache_resource
def get_models() -> Tuple[Yuan2_LLM, HuggingFaceEmbeddings]:
    llm = Yuan2_LLM(model_path)

    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return llm, embeddings
