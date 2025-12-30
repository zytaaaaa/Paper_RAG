import re
from typing import List, Optional, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Chunker:
    """Text chunking utilities (moved from app.py)"""
    def __init__(self, chunk_size=800, chunk_overlap=100):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", "；", "。", "！", "？", " ", ""],
            length_function=len
        )

    def _split_into_sections(self, text: str) -> list:
        lines = text.splitlines()
        heading_num_re = re.compile(r'^\s*(\d+(?:\.\d+)*\.?)+\s+(.{1,200})$')
        title_like_re = re.compile(r'^[A-Z][A-Za-z0-9\-\s]{1,120}$')
        citation_re = re.compile(r'\(\s*[^)]*\b\d{4}\b[^)]*\)|et\s+al\.', re.I)

        sections = []
        stack = []
        curr_lines = []

        def start_new_section():
            nonlocal curr_lines
            if curr_lines:
                sections.append({
                    "text": "\n".join(curr_lines).strip(),
                    "heading_hierarchy": [h for (_, h) in stack]
                })
            curr_lines = []

        for i, ln in enumerate(lines):
            s = ln.strip()
            if not s:
                curr_lines.append(ln)
                continue

            m = heading_num_re.match(s)
            is_heading = False
            is_citation = bool(citation_re.search(s))
            if m and not is_citation:
                num = m.group(1)
                title = m.group(2).strip()
                num_clean = num.rstrip('.')
                level = num_clean.count('.') + 1
                while stack and stack[-1][0] >= level:
                    stack.pop()
                start_new_section()
                stack.append((level, f"{num_clean} {title}"))
                curr_lines.append(s)
                is_heading = True
            else:
                is_allcaps = s.isupper() and len(s) <= 140 and len(s.split()) <= 10
                is_title_like = title_like_re.match(s) and not s.endswith('.') and len(s.split()) <= 10
                prev_blank = (i == 0) or (lines[i - 1].strip() == '')
                next_blank = (i + 1 == len(lines)) or (lines[i + 1].strip() == '')
                if not is_citation and (is_allcaps or (is_title_like and prev_blank and next_blank)) and len(s) < 200:
                    while stack and stack[-1][0] >= 1:
                        stack.pop()
                    start_new_section()
                    stack.append((1, s))
                    curr_lines.append(s)
                    is_heading = True

            if not is_heading:
                curr_lines.append(ln)

        if curr_lines:
            sections.append({
                "text": "\n".join(curr_lines).strip(),
                "heading_hierarchy": [h for (_, h) in stack]
            })

        if not sections:
            return [{"text": text, "heading_hierarchy": []}]

        processed = []
        for sec in sections:
            first_line = sec["text"].splitlines()[0].strip() if sec["text"].strip() else ''
            hier = list(sec.get("heading_hierarchy", []))
            if first_line:
                if heading_num_re.match(first_line) or (
                        first_line.isupper() and len(first_line.split()) <= 10) or title_like_re.match(first_line):
                    if not hier or hier[-1] != first_line:
                        hier = hier + [first_line]
            processed.append({"text": sec["text"], "heading_hierarchy": hier})
        return processed

    def _detect_heading_in_chunk(self, chunk_text: str) -> Optional[str]:
        for ln in chunk_text.splitlines()[:3]:
            s = ln.strip()
            if not s:
                continue
            if re.search(r'\(\s*[^)]*\b\d{4}\b[^)]*\)|et\s+al\.', s, re.I):
                break
            is_numbered = re.match(r'^\d+(?:\.\d+)*\.?\s+\S+', s)
            is_allcaps = s.isupper() and len(s) <= 140 and len(s.split()) <= 10
            is_title_like = re.match(r'^[A-Z][A-Za-z0-9\-\s]{1,120}$', s) and not s.endswith('.') and len(
                s.split()) <= 10
            if (is_numbered or is_allcaps or is_title_like) and len(s) < 200:
                return s
            break
        return None

    def _chunks_from_text(self, text: str) -> List[dict]:
        sections = self._split_into_sections(text)
        chunks = []
        global_i = 0
        for sec in sections:
            sec_text = sec.get("text", "")
            hierarchy = sec.get("heading_hierarchy", [])
            sub_texts = self.text_splitter.split_text(text=sec_text)
            merged_subs = []
            j = 0
            while j < len(sub_texts):
                cur = sub_texts[j]
                cur_strip = cur.strip()
                is_heading_like = False
                if cur_strip:
                    if re.match(r'^\d+(?:\.\d+)*\.?\s+\S+', cur_strip):
                        is_heading_like = True
                    elif cur_strip.isupper() and len(cur_strip.split()) <= 10:
                        is_heading_like = True
                    elif hierarchy and cur_strip == (hierarchy[-1] if hierarchy else ''):
                        is_heading_like = True
                if is_heading_like and j + 1 < len(sub_texts):
                    merged = cur + "\n\n" + sub_texts[j + 1]
                    merged_subs.append(merged)
                    j += 2
                    continue
                merged_subs.append(cur)
                j += 1

            for sub in merged_subs:
                heading = (hierarchy[-1] if hierarchy else None) or self._detect_heading_in_chunk(sub)
                metadata = {
                    "chunk_index": global_i,
                    "heading": heading,
                    "heading_hierarchy": hierarchy,
                    "heading_level": len(hierarchy)
                }
                chunks.append({"text": sub, "metadata": metadata})
                global_i += 1
        return chunks

    def list_chunks(self, docs, preview_chars: int = 200):
        text = ''.join([d.page_content for d in docs])
        chunks = self._chunks_from_text(text)
        out = []
        for c in chunks:
            i = c['metadata']['chunk_index']
            h = c['metadata'].get('heading')
            t = c['text']
            preview = t[:preview_chars].replace('\n', ' ')
            out.append({'index': i, 'heading': h, 'preview': preview, 'text': t, 'metadata': c['metadata']})
        return out

    def export_chunks(self, docs) -> str:
        text = ''.join([d.page_content for d in docs])
        chunks = self._chunks_from_text(text)
        return json.dumps(chunks, ensure_ascii=False, indent=2)
