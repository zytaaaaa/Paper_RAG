# 导入所需的库
import torch
import streamlit as st
st.set_page_config(page_title="Yuan2.0 学术论文助手", page_icon="📚", layout="wide")
from transformers import AutoTokenizer, AutoModelForCausalLM
# ✅ Prompts
from langchain_core.prompts import PromptTemplate

# ✅ Vector Stores (Chroma)
from langchain_community.vectorstores import Chroma

# ✅ Document Loaders
from langchain_community.document_loaders import PyPDFLoader

# ✅ Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# ✅ Chains
from langchain.chains import LLMChain  # 这个目前还能用，但推荐用 Runnable 替代
from langchain.chains.question_answering import load_qa_chain  # 仍可用，但逐步弃用

# ✅ Custom LLM
from langchain_core.language_models import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun

# ✅ Text Splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ✅ Schema
from langchain_core.documents import Document
import re
from collections import Counter
from typing import Any, List, Optional, Dict, TypedDict
import numpy as np

from rank_bm25 import BM25Okapi
import os
import hashlib
import json
import pickle
import io, csv
import requests
import xml.etree.ElementTree as ET
from functools import lru_cache
import time
# 向量模型下载
from modelscope import snapshot_download
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

# 取消实际下载（如需运行请取消注释，此处为示例）
#model_dir = snapshot_download('AI-ModelScope/bge-small-en-v1.5', cache_dir='./')
#model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='./')

# 定义模型路径（请根据实际下载路径修改）
model_path = './IEITYuan/Yuan2-2B-Mars-hf'
embedding_model_path = './AI-ModelScope/bge-small-en-v1___5'

# 定义模型数据类型
torch_dtype = torch.bfloat16  # A10


# torch_dtype = torch.float16 # P100

from ui import main


if __name__ == '__main__':
    main()
                  