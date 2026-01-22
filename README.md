# Local RAG Paper Assistant

这是一个基于本地大语言模型（LLM）的 RAG（检索增强生成）系统，专为学术论文阅读、摘要生成、多文档比较和文献检索而设计。

本项目利用本地部署的模型和 BGE Embedding 模型，确保数据隐私，同时提供强大的文档理解能力。前端采用 Streamlit 构建，操作简便。

## ✨ 主要功能

- **📄 论文智能问答**
  - 支持上传 PDF 论文。
  - 自动进行文本预处理（去除页眉、页脚、参考文献等）。
  - 自动生成论文摘要。
  - 基于混合检索（BM25 + 向量检索）的精准问答。

- **⚖️ 论文对比分析**
  - 支持同时上传两篇论文进行对比。
  - 使用 LangGraph 构建的智能 Agent 工作流。
  - 自动提取两篇论文的核心内容并生成对比报告。

- **🔍 外部文献搜索**
  - 集成 CrossRef 和 arXiv 搜索接口。
  - 支持基于文档内容自动提取关键词进行相关文献推荐。

- **🔒 本地化部署**
  - 模型全本地运行，无需依赖外部 API，保护数据隐私。
  - 支持 GPU 加速（需 NVIDIA 显卡）。

## 🛠️ 技术栈

- **LLM**: [IEITYuan/Yuan2-2B-Mars-hf](https://huggingface.co/IEITYuan/Yuan2-2B-Mars-hf)
- **Embedding**: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **Framework**: [LangChain](https://www.langchain.com/), [LangGraph](https://langchain-ai.github.io/langgraph/)
- **UI**: [Streamlit](https://streamlit.io/)
- **Vector Store**: Chroma
- **Retrieval**: BM25 + Vector Hybrid Retrieval

## 🚀 快速开始

### 1. 环境准备

建议使用 Python 3.10+ 环境。

```bash
# 克隆项目 (如果是从仓库下载)
# git clone <repository_url>
# cd RAG_project

# 安装依赖
pip install -r requirements.txt
```

> 注意：请根据您的 CUDA 版本安装对应的 `torch` 版本。

### 2. 模型下载

本项目需要下载以下两个模型到本地目录：

1.  **Yuan2-2B-Mars-hf**: 存放在 `IEITYuan/Yuan2-2B-Mars-hf`
2.  **bge-small-en-v1.5**: 存放在 `AI-ModelScope/bge-small-en-v1___5`

代码中已集成了 `modelscope` 下载逻辑，如需重新下载，可查看 `app.py` 中的相关注释。

### 3. 运行应用

```bash
streamlit run app.py
```

启动后，浏览器将自动打开 `http://localhost:8501`。

