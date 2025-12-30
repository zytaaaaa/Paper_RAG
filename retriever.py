import os
import json
import pickle
import re
import numpy as np
from typing import Any, List, Optional, Tuple
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from chunker import Chunker

class Retriever:
    """Retriever/Indexing utilities (moved from app.py ChatBot implementation)"""
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.chunker = Chunker()
        # indexes
        self.index_chunks: Optional[List[dict]] = None
        self.index_bm25 = None
        self.index_embeds = None
        self.chroma_store = None

    def _index_dir_for_hash(self, file_hash: str) -> str:
        return os.path.join("indexes", file_hash[:16])

    def build_index(self, docs, index_dir: str, overwrite: bool = False) -> bool:
        if os.path.exists(index_dir) and not overwrite:
            return self.load_index(index_dir)

        raw_text = ''.join([d.page_content for d in docs])
        chunks = self.chunker._chunks_from_text(raw_text)
        os.makedirs(index_dir, exist_ok=True)

        with open(os.path.join(index_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

        tokenized = [re.findall(r"\w+", c["text"].lower()) for c in chunks]
        with open(os.path.join(index_dir, "bm25.pkl"), "wb") as f:
            pickle.dump(tokenized, f)

        try:
            bm25 = BM25Okapi(tokenized)
            with open(os.path.join(index_dir, "bm25_obj.pkl"), "wb") as f:
                pickle.dump(bm25, f)
            self.index_bm25 = bm25
        except Exception:
            self.index_bm25 = None

        try:
            embeds = self.embeddings.embed_documents([c["text"] for c in chunks])
            np.save(os.path.join(index_dir, "embeddings.npy"), np.array(embeds))
            self.index_embeds = np.array(embeds)
        except Exception:
            self.index_embeds = None

        try:
            texts = [c["text"] for c in chunks]
            metadatas = [c.get("metadata", {}) for c in chunks]
            chroma_store = Chroma.from_texts(texts, embedding=self.embeddings, metadatas=metadatas,
                                             persist_directory=index_dir, collection_name="docs")
            try:
                chroma_store.persist()
            except Exception:
                pass
            self.chroma_store = chroma_store
        except Exception:
            self.chroma_store = None

        self.index_chunks = chunks
        return True

    def load_index(self, index_dir: str) -> bool:
        if not os.path.exists(index_dir):
            return False

        try:
            with open(os.path.join(index_dir, "chunks.json"), "r", encoding="utf-8") as f:
                all_chunks = json.load(f)
            if isinstance(all_chunks, list) and all_chunks and isinstance(all_chunks[0], str):
                chunks = []
                for i, t in enumerate(all_chunks):
                    heading = self.chunker._detect_heading_in_chunk(t)
                    chunks.append({"text": t, "metadata": {"chunk_index": i, "heading": heading}})
                self.index_chunks = chunks
            else:
                self.index_chunks = all_chunks
        except Exception:
            self.index_chunks = None

        try:
            with open(os.path.join(index_dir, "bm25_obj.pkl"), "rb") as f:
                self.index_bm25 = pickle.load(f)
        except Exception:
            try:
                with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
                    tokenized = pickle.load(f)
                self.index_bm25 = BM25Okapi(tokenized) if tokenized else None
            except Exception:
                self.index_bm25 = None

        try:
            self.index_embeds = np.load(os.path.join(index_dir, "embeddings.npy"))
        except Exception:
            self.index_embeds = None

        try:
            chroma_store = Chroma(persist_directory=index_dir, embedding_function=self.embeddings,
                                  collection_name="docs")
            self.chroma_store = chroma_store
        except Exception:
            self.chroma_store = None

        return True

    def list_chunks(self, docs, preview_chars: int = 200):
        return self.chunker.list_chunks(docs, preview_chars=preview_chars)

    def _retrieve_indices(self, chunks, query, method='rrf', top_k=20, rrf_k=60):
        n = len(chunks)
        bm25_ranks = np.full(n, n + 1, dtype=int)
        tokenized = [re.findall(r"\w+", c["text"].lower()) for c in chunks]
        if BM25Okapi is not None and any(tokenized):
            bm25 = BM25Okapi(tokenized)
            q_tokens = re.findall(r"\w+", query.lower())
            bm25_scores = np.array(bm25.get_scores(q_tokens), dtype=float)
            bm_idx = np.argsort(-bm25_scores)
            bm25_ranks[bm_idx] = np.arange(1, n + 1)

        emb_ranks = np.full(n, n + 1, dtype=int)
        emb_sims = np.zeros(n, dtype=float)
        try:
            if self.chroma_store is not None:
                docs_sim = self.chroma_store.similarity_search(query=query, k=n)
            else:
                texts = [c["text"] for c in chunks]
                metadatas = [c.get("metadata", {}) for c in chunks]
                docs_sim = Chroma.from_texts(texts, embedding=self.embeddings, metadatas=metadatas,
                                             collection_name="tmp").similarity_search(query=query, k=n)
            for rank, doc in enumerate(docs_sim, start=1):
                chunk_idx = doc.metadata.get("chunk_index")
                if chunk_idx is not None and 0 <= int(chunk_idx) < n:
                    emb_ranks[int(chunk_idx)] = rank
                    score = getattr(doc, "score", None) or doc.metadata.get("score", None)
                    if score is not None:
                        try:
                            emb_sims[int(chunk_idx)] = float(score)
                        except Exception:
                            pass
        except Exception:
            try:
                chunk_embeds = self.embeddings.embed_documents([c["text"] for c in chunks])
                q_embed = self.embeddings.embed_query(query)
                if q_embed is not None and len(chunk_embeds) == n:
                    q = np.array(q_embed)
                    emb_sims = np.array([float(np.dot(q, np.array(e))) for e in chunk_embeds], dtype=float)
                    emb_idx = np.argsort(-emb_sims)
                    emb_ranks[emb_idx] = np.arange(1, n + 1)
            except Exception:
                pass

        if method == 'bm25':
            return list(np.argsort(bm25_ranks)[:top_k])
        if method == 'emb':
            return list(np.argsort(-emb_sims)[:top_k])

        rrf_scores = np.zeros(n, dtype=float)
        rrf_scores += 1.0 / (rrf_k + bm25_ranks)
        rrf_scores += 1.0 / (rrf_k + emb_ranks)
        return list(np.argsort(-rrf_scores)[:top_k])

    def run(self, docs, query, history, top_k=5, rrf_k=60) -> Tuple[List[Document], str]:
        text = ''.join([doc.page_content for doc in docs])
        chunks = self.chunker._chunks_from_text(text)
        n = len(chunks)

        tokenized = [re.findall(r"\w+", c["text"].lower()) for c in chunks]
        bm25_ranks = np.full(n, n + 1, dtype=int)
        if BM25Okapi is not None and any(tokenized):
            bm25 = BM25Okapi(tokenized)
            q_tokens = re.findall(r"\w+", query.lower())
            bm25_scores = np.array(bm25.get_scores(q_tokens), dtype=float)
            bm_idx = np.argsort(-bm25_scores)
            bm25_ranks[bm_idx] = np.arange(1, n + 1)

        emb_ranks = np.full(n, n + 1, dtype=int)
        try:
            if self.chroma_store is not None:
                docs_sim = self.chroma_store.similarity_search(query=query, k=n)
            else:
                docs_sim = Chroma.from_texts([c["text"] for c in chunks], embedding=self.embeddings,
                                             collection_name="tmp").similarity_search(query=query, k=n)
            for rank, doc in enumerate(docs_sim, start=1):
                chunk_idx = doc.metadata.get("chunk_index")
                if chunk_idx is not None and 0 <= int(chunk_idx) < n:
                    emb_ranks[int(chunk_idx)] = rank
        except Exception:
            try:
                if self.chroma_store is not None:
                    VectorStore = self.chroma_store
                else:
                    VectorStore = Chroma.from_texts([c["text"] for c in chunks], embedding=self.embeddings,
                                                    collection_name="tmp")
                docs_sim = VectorStore.similarity_search(query=query, k=top_k)
            except Exception:
                docs_sim = [Document(page_content=c["text"], metadata=c.get("metadata", {})) for c in chunks[:top_k]]
            response = self.llm(self._format_chain_input(docs_sim, history, query))
            return docs_sim, response

        rrf_scores = np.zeros(n, dtype=float)
        rrf_scores += 1.0 / (rrf_k + bm25_ranks)
        rrf_scores += 1.0 / (rrf_k + emb_ranks)

        final_idxs = list(np.argsort(-rrf_scores)[:top_k])
        selected_docs = [Document(page_content=chunks[int(i)]["text"], metadata=chunks[int(i)]["metadata"]) for i in final_idxs]

        response = self.llm(self._format_chain_input(selected_docs, history, query))
        return selected_docs, response

    def _format_chain_input(self, docs, history, question):
        # Simple prompt formatting: the original used a langchain chain; keep compatibility by creating a text prompt
        history_text = history or ""
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"Context:\n{context[:4000]}\n\nHistory:\n{history_text}\n\nQuestion:\n{question}"
        return prompt
