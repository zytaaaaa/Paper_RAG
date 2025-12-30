from typing import Any, List, Optional, Dict, TypedDict
import os
import hashlib
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from retriever import Retriever
from summarizer import Summarizer
from cleaner import clean_documents

class PaperCompareState(TypedDict):
    paper1_file: Optional[bytes]
    paper2_file: Optional[bytes]
    paper1_docs: Optional[List[Document]]
    paper2_docs: Optional[List[Document]]
    paper1_index: Optional[Any]
    paper2_index: Optional[Any]
    paper1_core_content: Optional[Dict[str, str]]
    paper2_core_content: Optional[Dict[str, str]]
    comparison_result: Optional[str]


class PaperCompareNodes:
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.retriever = Retriever(llm, embeddings)
        self.summarizer = Summarizer(llm)

    def load_papers(self, state: PaperCompareState) -> PaperCompareState:
        st.write("🔍 正在加载并预处理论文...")
        paper1_file = state["paper1_file"]
        temp1 = "temp1.pdf"
        with open(temp1, "wb") as f:
            f.write(paper1_file)
        loader1 = PyPDFLoader(temp1)
        paper1_docs = loader1.load()
        paper1_docs = clean_documents(paper1_docs)

        paper2_file = state["paper2_file"]
        temp2 = "temp2.pdf"
        with open(temp2, "wb") as f:
            f.write(paper2_file)
        loader2 = PyPDFLoader(temp2)
        paper2_docs = loader2.load()
        paper2_docs = clean_documents(paper2_docs)

        state["paper1_docs"] = paper1_docs
        state["paper2_docs"] = paper2_docs
        st.success("✅ 两篇论文加载并预处理完成！")
        return state

    def build_indexes(self, state: PaperCompareState) -> PaperCompareState:
        st.write("📚 正在为论文构建检索索引...")
        paper1_docs = state["paper1_docs"]
        paper2_docs = state["paper2_docs"]

        paper1_hash = hashlib.sha256(b"".join([d.page_content.encode() for d in paper1_docs])).hexdigest()
        paper1_index_dir = os.path.join("indexes", paper1_hash[:16])
        if not os.path.exists(paper1_index_dir):
            self.retriever.build_index(paper1_docs, paper1_index_dir)
        self.retriever.load_index(paper1_index_dir)
        state["paper1_index"] = paper1_index_dir

        paper2_hash = hashlib.sha256(b"".join([d.page_content.encode() for d in paper2_docs])).hexdigest()
        paper2_index_dir = os.path.join("indexes", paper2_hash[:16])
        if not os.path.exists(paper2_index_dir):
            self.retriever.build_index(paper2_docs, paper2_index_dir)
        self.retriever.load_index(paper2_index_dir)
        state["paper2_index"] = paper2_index_dir

        st.success("✅ 两篇论文索引构建完成！")
        return state

    def retrieve_core_content(self, state: PaperCompareState) -> PaperCompareState:
        st.write("🔍 正在检索论文核心内容...")
        paper1_docs = state["paper1_docs"]
        paper2_docs = state["paper2_docs"]

        core_questions = [
            "What is the research question of this paper?",
            "What methods are used in this paper?",
            "What are the experimental results of this paper?",
            "What is the conclusion of this paper?"
        ]

        paper1_core = {}
        for q in core_questions:
            chunks, response = self.retriever.run(paper1_docs, q, history="", top_k=3)
            paper1_core[q] = response
        state["paper1_core_content"] = paper1_core

        paper2_core = {}
        for q in core_questions:
            chunks, response = self.retriever.run(paper2_docs, q, history="", top_k=3)
            paper2_core[q] = response
        state["paper2_core_content"] = paper2_core

        st.success("✅ 论文核心内容检索完成！")
        return state

    def compare_and_summarize(self, state: PaperCompareState) -> PaperCompareState:
        st.write("📝 正在对比两篇论文并总结异同点...")
        paper1_core = state["paper1_core_content"]
        paper2_core = state["paper2_core_content"]

        compare_prompt = f"""
        你是一名资深的学术研究员，擅长对比分析学术论文。请根据以下两篇论文的核心内容，从**研究问题、研究方法、实验结果、结论**四个维度，**详细分析它们的异同点**，要求：
        1. 每个维度需分别列出**相同点**和**不同点**，并结合论文内容**举例说明**；
        2. 若某一维度无相同点/不同点，需明确说明“无”，并解释原因；
        3. 输出格式为Markdown，使用## 分维度，### 分相同点/不同点，用- 列出具体内容；
        4. 语言正式、逻辑严谨，每个分析点至少50字。

        论文1核心内容：
        {paper1_core}

        论文2核心内容：
        {paper2_core}

        请输出对比结果：
        """

        response = self.llm(compare_prompt)
        state["comparison_result"] = response
        st.success("✅ 论文异同点总结完成！")
        return state

    def final_output(self, state: PaperCompareState) -> PaperCompareState:
        st.subheader("📊 两篇论文异同点对比结果")
        st.markdown(state["comparison_result"])
        return state


def build_paper_compare_agent(llm, embeddings) -> CompiledStateGraph:
    nodes = PaperCompareNodes(llm, embeddings)
    graph = StateGraph(PaperCompareState)
    graph.add_node("load_papers", nodes.load_papers)
    graph.add_node("build_indexes", nodes.build_indexes)
    graph.add_node("retrieve_core_content", nodes.retrieve_core_content)
    graph.add_node("compare_and_summarize", nodes.compare_and_summarize)
    graph.add_node("final_output", nodes.final_output)
    graph.add_edge("load_papers", "build_indexes")
    graph.add_edge("build_indexes", "retrieve_core_content")
    graph.add_edge("retrieve_core_content", "compare_and_summarize")
    graph.add_edge("compare_and_summarize", "final_output")
    graph.add_edge("final_output", END)
    graph.set_entry_point("load_papers")
    compiled_graph = graph.compile()
    return compiled_graph
