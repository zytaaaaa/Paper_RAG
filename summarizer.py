from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List, Optional
import re

summarizer_template = """
假设你是一个AI科研助手，请用一段话概括下面文章的主要内容，200字左右。

{text}
"""

class Summarizer:
    """Summarizer wrapper (moved from app.py)"""

    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template=summarizer_template
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def extract_abstract_from_pages(self, docs, max_pages=3):
        abstract_hdr_re = re.compile(r'(?:^|\n)\s*(ABSTRACT|Abstract|摘要|SUMMARY)\s*[:\n]', re.I)
        next_section_re = re.compile(r'(?:^|\n)\s*([A-Z][A-Z\s]{2,}|KEYWORDS|Key words|KEY WORDS|关键词)\s*[:\n]', re.I)

        for doc in docs[:max_pages]:
            text = doc.page_content
            m = abstract_hdr_re.search(text)
            if m:
                start = m.end()
                nxt = next_section_re.search(text, start)
                end = nxt.start() if nxt else start + 2000
                candidate = text[start:end].strip()
                if len(candidate) > 40:
                    return candidate
        return None

    def fallback_first_page(self, docs, chars=800):
        first = docs[0].page_content
        paragraphs = [p.strip() for p in first.split('\n\n') if p.strip()]
        if paragraphs:
            candidate = '\n\n'.join(paragraphs[:2])
            return candidate[:chars]
        return first[:chars]

    def summarize(self, docs, strategy='auto'):
        if strategy in ('strict', 'auto'):
            abstract = self.extract_abstract_from_pages(docs)
            if abstract:
                return self.chain.run(abstract)

        if strategy in ('auto', 'llm'):
            fallback = self.fallback_first_page(docs)
            prompt = f"请用一段话（200字左右）概括下面文章的主要内容：\n\n{fallback}"
            return self.chain.run(prompt)

        full_text = '\n'.join([d.page_content for d in docs])
        prompt = f"请用一段话（200字左右）概括下面文章的主要内容（如果有Abstract优先使用）：\n\n{full_text[:4000]}"
        return self.chain.run(prompt)
