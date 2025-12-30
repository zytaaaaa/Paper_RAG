import re
from collections import Counter
from langchain_core.documents import Document
from typing import List


def clean_documents(docs, top_n=5, bottom_n=5, repeat_threshold=0.4) -> List[Document]:
    """Clean pages: remove headers/footers, reference section and noisy metadata."""
    top_lines = []
    bottom_lines = []
    for doc in docs:
        lines = [l.strip() for l in doc.page_content.splitlines() if l.strip()]
        top_lines.extend(lines[:top_n])
        bottom_lines.extend(lines[-bottom_n:])

    counter = Counter(top_lines + bottom_lines)
    repeated = set([
        line for line, cnt in counter.items()
        if cnt >= max(2, int(len(docs) * repeat_threshold)) and len(line) < 200
    ])

    page_re = re.compile(r'^(page\s*\d+|\d+\s*$|第\s*\d+\s*页|pp\.\s*\d+|\d+\s*/\s*\d+)$', re.I)
    journal_keywords = [
        'journal', 'proceedings', 'ieee', 'acm', 'springer', 'elsevier',
        'nature', 'science', 'cell', 'arxiv', 'preprint', 'conference',
        'transactions', 'symposium', 'workshop', 'meeting', 'volume',
        'vol.', 'vol', 'issue', 'no.', 'number', 'pp.', 'pages',
        '期刊', '出版社', '卷', '期', 'issn', 'isbn', 'doi',
        'received', 'revised', 'accepted', 'published', 'copyright',
        'all rights reserved', 'rights reserved', '©', 'peer review'
    ]
    reference_section_patterns = [
        r'^\s*references?\s*[:]?'
        r'^\s*bibliography\s*[:]?'
        r'^\s*参考文献\s*[:]?'
        r'^\s*引用文献\s*[:]?'
        r'^\s*主要参考文献\s*[:]?'
    ]
    reference_section_patterns = [
        r'^\s*references?\s*[:]?[\s$]*',
        r'^\s*bibliography\s*[:]?[\s$]*',
        r'^\s*参考文献\s*[:]?[\s$]*',
        r'^\s*引用文献\s*[:]?[\s$]*',
        r'^\s*主要参考文献\s*[:]?[\s$]*',
    ]
    reference_patterns = [
        r'^\[\d+\]',
        r'^\d+\.\s',
        r'^\(\d{4}\)',
        r'^[A-Z][a-z]+ et al\.',
        r'^[A-Z][a-z]+, [A-Z]\\.',
        r'^\d{4}\.',
        r'^[A-Z]\\.[A-Z]\\.',
        r'^[A-Z][a-zÀ-ÿ]+(?:-[A-Z][a-zÀ-ÿ]+)*,\s*[A-Z]\\.?[A-Z]\\?\.?',
    ]

    def find_reference_section(lines):
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if any(re.match(pattern, line_stripped, re.I) for pattern in reference_section_patterns):
                return i
        return -1

    def is_reference_line(line):
        line_stripped = line.strip()
        if not line_stripped:
            return False
        for pattern in reference_patterns:
            if re.match(pattern, line_stripped):
                return True
        ref_indicators = ['et al.', 'pp.', 'vol.', 'no.', 'issue',
                          'In:', 'Proc.', 'Conf.', 'Symp.', 'Workshop',
                          'arXiv:', 'doi:', 'https://doi.org/', 'http://']
        if any(indicator in line_stripped for indicator in ref_indicators):
            return True
        author_year_pattern = r'^[A-Z][a-z]+\s+[A-Z]\\.?\s*\(\d{4}\)'
        if re.match(author_year_pattern, line_stripped):
            return True
        return False

    cleaned_docs = []
    for doc_idx, doc in enumerate(docs):
        lines = doc.page_content.splitlines()
        ref_start = find_reference_section(lines)
        new_lines = []
        in_reference_section = False

        for i, line in enumerate(lines):
            s = line.strip()
            if not s:
                new_lines.append('')
                continue
            if ref_start >= 0 and i >= ref_start:
                in_reference_section = True
            if in_reference_section:
                continue
            if page_re.match(s):
                continue
            low = s.lower()
            if any(k in low for k in journal_keywords) and len(s) < 200:
                if len(s) < 100 or any(k in low for k in ['figure', 'table', 'equation', 'algorithm']):
                    continue
            if s in repeated:
                continue
            if len(s) <= 4 and s.isdigit():
                continue
            metadata_patterns = [
                r'^[\w\s]+ manuscript\s*$',
                r'^this is the author\'s version\s*$',
                r'^final published version\s*$',
                r'^submitted to\s*[\w\s]+$',
                r'^under review\s*$',
            ]
            if any(re.match(pattern, s, re.I) for pattern in metadata_patterns):
                continue
            new_lines.append(line)

        cleaned_content = '\n'.join(new_lines)
        cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
        cleaned_doc = Document(
            page_content=cleaned_content,
            metadata={
                **doc.metadata,
                'original_page': doc_idx + 1,
                'has_references_removed': in_reference_section,
                'cleaned': True
            }
        )
        cleaned_docs.append(cleaned_doc)
    return cleaned_docs


def extract_search_query_from_docs(docs, summarizer=None, top_k_keywords=8):
    try:
        first_page = docs[0].page_content
        for ln in first_page.splitlines():
            s = ln.strip()
            if s and len(s) <= 200:
                title_candidate = s
                break
        else:
            title_candidate = ''
    except Exception:
        title_candidate = ''

    summary_text = ''
    try:
        if summarizer:
            summary_text = summarizer.summarize(docs, strategy='auto')
    except Exception:
        try:
            summary_text = docs[0].page_content[:800]
        except Exception:
            summary_text = ''

    text = (title_candidate + '\n' + summary_text).lower()
    tokens = re.findall(r"\w+", text)
    stopwords = set(["the","and","of","in","to","a","we","for","with","is","on","by","this","that","our","using","use","based","are","an","from","as","be","may","these","which"])
    freq = Counter([t for t in tokens if len(t) > 3 and not t.isdigit() and t not in stopwords])
    keywords = [w for w,_ in freq.most_common(top_k_keywords)]
    if keywords:
        return ' '.join(keywords)
    if title_candidate:
        return title_candidate
    return (summary_text or text)[:500]
