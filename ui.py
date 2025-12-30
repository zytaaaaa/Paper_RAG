import streamlit as st
import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from models import get_models
from retriever import Retriever
from summarizer import Summarizer
from external_search import fetch_crossref, fetch_arxiv, cache_save, cache_load, is_recommend_command, parse_recommend_command
from agents import build_paper_compare_agent
from cleaner import clean_documents, extract_search_query_from_docs
import os

MAX_HISTORY_ROUNDS = 5


def single_doc_qa_tab(llm, embeddings):
    st.subheader("📄 单篇论文智能问答")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "summary" not in st.session_state:
        st.session_state.summary = ""

    summarizer = Summarizer(llm)
    retriever = Retriever(llm, embeddings)

    uploaded_file = st.file_uploader("上传你的论文PDF文件", type='pdf', key="single_doc_uploader")

    if uploaded_file:
        file_content = uploaded_file.read()
        temp_file_path = "temp_single.pdf"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_content)
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        cleaning = st.checkbox("启用预处理（去除页眉页脚/参考文献）", value=True, key="single_doc_clean")
        if cleaning:
            docs = clean_documents(docs)
            st.success("已完成论文预处理")

        if not st.session_state.summary:
            with st.spinner("正在生成论文摘要..."):
                summary = summarizer.summarize(docs, strategy='auto')
                st.session_state.summary = summary
        else:
            summary = st.session_state.summary

        st.subheader("📝 论文摘要")
        edited_summary = st.text_area("编辑摘要", value=summary, height=200, key="single_doc_summary")
        if st.button("更新摘要", key="update_summary"):
            st.session_state.summary = edited_summary
            summary = edited_summary
        st.info(summary)

        with st.expander("📑 查看论文切块详情", expanded=False):
            chunks = retriever.list_chunks(docs, preview_chars=300)
            for i, chunk in enumerate(chunks):
                st.markdown(f"### Chunk {i}")
                st.write(f"**元数据**：")
                st.write(f"- 全局索引：{chunk['metadata'].get('chunk_index', '无')}")
                st.write(f"- 标题层级：{chunk['metadata'].get('heading_hierarchy', [])}")
                st.write(f"- 标题级别：{chunk['metadata'].get('heading_level', 0)}")
                st.write(f"- 标题内容：{chunk['heading'] or '无'}")
                st.write(f"**完整内容**：")
                st.text_area(label=f"Chunk {i} 完整文本", value=chunk['text'], height=300, key=f"chunk_{i}_content")
                st.divider()

        file_hash = hashlib.sha256(file_content).hexdigest()
        index_dir = retriever._index_dir_for_hash(file_hash)
        if os.path.exists(index_dir):
            loaded = retriever.load_index(index_dir)
            if loaded:
                st.info(f"已加载本地索引：{index_dir}")
            else:
                retriever.build_index(docs, index_dir, overwrite=True)
        else:
            retriever.build_index(docs, index_dir)
            st.success(f"已构建论文索引：{index_dir}")

        st.subheader("🔗 推荐相关文献")
        col_a, col_b, col_c = st.columns([2,1,1])
        with col_a:
            source = st.selectbox("选择文献来源", ["CrossRef", "arXiv"], key="rec_source")
        with col_b:
            topk = st.slider("返回数量", min_value=3, max_value=20, value=5, key="rec_topk")
        with col_c:
            rec_button = st.button("🔍 推荐相关文献", key="rec_button")

        if rec_button:
            with st.spinner("正在查询外部知识库..."):
                query_text = extract_search_query_from_docs(docs, summarizer)
                cache_dir = index_dir
                cached = cache_load(cache_dir, source)
                if cached and isinstance(cached, list) and len(cached) >= 1:
                    recs = cached
                else:
                    if source == "CrossRef":
                        recs = fetch_crossref(query_text, topk)
                    else:
                        recs = fetch_arxiv(query_text, topk)
                    cache_save(cache_dir, source, recs)

            if recs:
                st.success(f"为你的论文找到 {len(recs)} 条可能相关的文献（来源：{source}）")
                for i, r in enumerate(recs, start=1):
                    title = r.get('title','')
                    authors = r.get('authors','')
                    venue = r.get('venue','') or r.get('container','') or ''
                    year = r.get('year','')
                    url = r.get('url','') or r.get('link','') or ''
                    doi = r.get('doi','')
                    st.markdown(f"**{i}. {title}**")
                    st.write(f"- **作者**：{authors}")
                    st.write(f"- **期刊/来源**：{venue}  - **年份**：{year}")
                    if doi:
                        st.write(f"- **DOI**：{doi}  - **链接**：{url}")
                    else:
                        st.write(f"- **链接**：{url}")
                    st.divider()
            else:
                st.warning("未找到相关文献，请尝试更改来源或增加返回数量。")

        st.subheader("💬 论文问答区")
        for user_msg, assistant_msg in st.session_state.chat_history:
            st.chat_message("user").write(user_msg)
            st.chat_message("assistant").write(assistant_msg)

        query = st.text_input("请输入你的问题（支持多轮对话，或输入“推荐相关文献”以触发文献推荐）", key="single_doc_query")
        if query:
            st.chat_message("user").write(query)
            history_str = ""
            recent_history = st.session_state.chat_history[-MAX_HISTORY_ROUNDS:]
            start_idx = len(st.session_state.chat_history) - len(recent_history) + 1
            for idx, (u, a) in enumerate(recent_history, start_idx):
                history_str += f"轮次{idx}：\n用户：{u}\n助手：{a}\n\n"
            history_str += f"轮次{len(st.session_state.chat_history) + 1}：\n用户：{query}\n"

            if is_recommend_command(query):
                cmd_source, cmd_topk = parse_recommend_command(query)
                effective_source = cmd_source or source
                effective_topk = cmd_topk or topk
                with st.spinner("正在查询外部知识库以推荐相关文献..."):
                    query_text = extract_search_query_from_docs(docs, summarizer)
                    cache_dir = index_dir
                    cached = cache_load(cache_dir, effective_source)
                    if cached and isinstance(cached, list) and len(cached) >= 1:
                        recs = cached[:effective_topk]
                    else:
                        if effective_source == "CrossRef":
                            recs = fetch_crossref(query_text, effective_topk)
                        else:
                            recs = fetch_arxiv(query_text, effective_topk)
                        cache_save(cache_dir, effective_source, recs)
                if recs:
                    st.success(f"为你的论文找到 {len(recs[:effective_topk])} 条可能相关的文献（来源：{effective_source}）")
                    for i, r in enumerate(recs[:effective_topk], start=1):
                        title = r.get('title','')
                        authors = r.get('authors','')
                        venue = r.get('venue','') or r.get('container','') or ''
                        year = r.get('year','')
                        url = r.get('url','') or r.get('link','') or ''
                        doi = r.get('doi','')
                        st.markdown(f"**{i}. {title}**")
                        st.write(f"- **作者**：{authors}")
                        st.write(f"- **期刊/来源**：{venue}  \n- **年份**：{year}")
                        if doi:
                            st.write(f"- **DOI**：{doi}  \n- **链接**：{url}")
                        else:
                            st.write(f"- **链接**：{url}")
                        st.divider()
                    response = f"已推荐 {len(recs[:effective_topk])} 条文献（来源：{effective_source}）"
                else:
                    st.warning("未找到相关文献，请尝试更改来源或增加返回数量。")
                    response = "未找到相关文献"
                st.chat_message("assistant").write(response)
                st.session_state.chat_history.append((query, response))
            else:
                with st.spinner("正在检索并生成回答..."):
                    chunks, response = retriever.run(docs, query, history_str)
                st.chat_message("assistant").write("**检索到的相关内容**：")
                st.write([c.page_content[:200] + "..." for c in chunks])
                st.chat_message("assistant").write("**回答**：")
                st.write(response)
                st.session_state.chat_history.append((query, response))

        if st.button("🗑️ 清空对话历史", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()


def paper_compare_tab(llm, embeddings):
    st.subheader("🔍 两篇论文对比分析")
    agent = build_paper_compare_agent(llm, embeddings)

    col1, col2 = st.columns(2)
    with col1:
        paper1_file = st.file_uploader("上传第一篇论文", type="pdf", key="paper1_uploader")
    with col2:
        paper2_file = st.file_uploader("上传第二篇论文", type="pdf", key="paper2_uploader")

    if paper1_file and paper2_file:
        paper1_bytes = paper1_file.read()
        paper2_bytes = paper2_file.read()

        initial_state = {
            'paper1_file': paper1_bytes,
            'paper2_file': paper2_bytes,
            'paper1_docs': None,
            'paper2_docs': None,
            'paper1_index': None,
            'paper2_index': None,
            'paper1_core_content': None,
            'paper2_core_content': None,
            'comparison_result': None
        }

        if st.button("🚀 开始对比分析", key="start_compare"):
            with st.spinner("Agent正在处理，请稍候..."):
                final_state = agent.invoke(initial_state)

        if st.button("🗑️ 清空上传文件", key="clear_papers"):
            st.rerun()


def main():

    st.title("📚 Yuan2.0 学术论文智能助手")

    with st.spinner("正在加载模型和嵌入..."):
        llm, embeddings = get_models()
    st.success("✅ 模型加载完成！")

    tab1, tab2 = st.tabs(["📄 单篇论文问答", "🔍 两篇论文对比"])
    with tab1:
        single_doc_qa_tab(llm, embeddings)
    with tab2:
        paper_compare_tab(llm, embeddings)


if __name__ == '__main__':
    main()
