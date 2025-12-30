import os
import json
import requests
import xml.etree.ElementTree as ET
from functools import lru_cache
from typing import List
import re


def cache_save(index_dir, source, data):
    try:
        os.makedirs(index_dir, exist_ok=True)
        with open(os.path.join(index_dir, f"recs_{source}.json"), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def cache_load(index_dir, source):
    try:
        p = os.path.join(index_dir, f"recs_{source}.json")
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return None


# CrossRef 查询
@lru_cache(maxsize=128)
def fetch_crossref(query: str, top_k: int = 5) -> List[dict]:
    try:
        params = {'query': query, 'rows': top_k}
        r = requests.get('https://api.crossref.org/works', params=params, timeout=10)
        r.raise_for_status()
        items = r.json().get('message', {}).get('items', [])
        results = []
        for it in items:
            title = ' '.join(it.get('title', [])) if it.get('title') else ''
            authors = []
            for a in it.get('author', [])[:5]:
                name_parts = []
                if a.get('given'):
                    name_parts.append(a.get('given'))
                if a.get('family'):
                    name_parts.append(a.get('family'))
                authors.append(' '.join(name_parts))
            venue = ' '.join(it.get('container-title', [])) if it.get('container-title') else ''
            year = ''
            if it.get('published-print') and it['published-print'].get('date-parts'):
                year = str(it['published-print']['date-parts'][0][0])
            elif it.get('published-online') and it['published-online'].get('date-parts'):
                year = str(it['published-online']['date-parts'][0][0])
            doi = it.get('DOI')
            url = it.get('URL') or (f"https://doi.org/{doi}" if doi else '')
            results.append({'title': title, 'authors': ', '.join(authors), 'venue': venue, 'year': year, 'doi': doi, 'url': url})
        return results
    except Exception:
        return []


# arXiv 查询
@lru_cache(maxsize=128)
def fetch_arxiv(query: str, top_k: int = 5) -> List[dict]:
    try:
        params = {'search_query': f'all:{query}', 'start': 0, 'max_results': top_k}
        r = requests.get('http://export.arxiv.org/api/query', params=params, timeout=10)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', ns)
        results = []
        for e in entries[:top_k]:
            title = ''.join(e.find('atom:title', ns).text.split()) if e.find('atom:title', ns) is not None else ''
            authors = []
            for a in e.findall('atom:author', ns):
                name = a.find('atom:name', ns).text if a.find('atom:name', ns) is not None else ''
                authors.append(name)
            published = e.find('atom:published', ns).text if e.find('atom:published', ns) is not None else ''
            link = ''
            for l in e.findall('atom:link', ns):
                if l.get('rel') == 'alternate':
                    link = l.get('href')
                    break
            summary = e.find('atom:summary', ns).text if e.find('atom:summary', ns) is not None else ''
            results.append({'title': title, 'authors': ', '.join(authors), 'venue': 'arXiv', 'year': published[:4], 'url': link, 'summary': summary})
        return results
    except Exception:
        return []


def is_recommend_command(q: str) -> bool:
    ql = q.lower()
    patterns = ['推荐', '推荐相关', '推荐文献', '找相关', '相关文献', 'recommend', 'find related', 'find references']
    return any(p in ql for p in patterns)


def parse_recommend_command(q: str):
    ql = q.lower()
    source = None
    if 'arxiv' in ql or 'arx' in ql or '预印本' in ql:
        source = 'arXiv'
    elif 'crossref' in ql or 'doi' in ql:
        source = 'CrossRef'
    topk = None
    m = re.search(r'(?:topk|n|数量|返回)\s*[:=]?\s*(\d{1,2})', ql)
    if m:
        topk = int(m.group(1))
    else:
        m2 = re.search(r'\b(\d{1,2})\b\s*(?:条|个|results?)', ql)
        if m2:
            topk = int(m2.group(1))
    return source, topk
