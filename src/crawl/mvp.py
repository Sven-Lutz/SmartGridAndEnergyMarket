# src/crawl/mvp.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import deque
from typing import Iterable, List, Optional, Tuple

import time
import json
import yaml
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin


@dataclass(frozen=True)
class MVPLimits:
    max_depth: int
    max_docs_per_municipality: int
    max_pdfs_per_municipality: int
    max_total_requests: int
    rate_limit_seconds: float


def _host(url: str) -> str:
    return urlparse(url).netloc.lower()


def _allowed_domain(url: str, allowed_domains: List[str]) -> bool:
    if not allowed_domains:
        return True
    h = _host(url)
    return any(h == d.lower() for d in allowed_domains)


def _kw_gate(text: str, keywords: List[str]) -> bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in keywords)


def _load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _extract_links_html(base_url: str, html: bytes) -> Iterable[Tuple[str, str]]:
    """
    Yields (abs_url, anchor_text)
    Pylance-safe: href must be str.
    """
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a"):
        href = a.get("href")
        if not isinstance(href, str):
            continue
        text = a.get_text(" ", strip=True)
        abs_url = urljoin(base_url, href)
        yield abs_url, text


def run_mvp(dataset_path: str, mvp_config_path: str, out_root: str) -> None:
    """
    dataset_path: configs/datasets/mvp_bayern.yaml (contains municipalities + maybe limits/filters)
    mvp_config_path: optional second config (if you still keep it separate)
    out_root: e.g. data/de_by/staging/mvp_bayern
    """
    ds = _load_yaml(dataset_path)
    cfg = _load_yaml(mvp_config_path) if mvp_config_path else {}

    limits_cfg = (cfg.get("limits") or ds.get("limits") or {})
    filters_cfg = (cfg.get("filters") or ds.get("filters") or {})

    limits = MVPLimits(
        max_depth=int(limits_cfg.get("max_depth", 2)),
        max_docs_per_municipality=int(limits_cfg.get("max_docs_per_municipality", limits_cfg.get("max_pages_per_municipality", 40))),
        max_pdfs_per_municipality=int(limits_cfg.get("max_pdfs_per_municipality", 8)),
        max_total_requests=int(limits_cfg.get("max_total_requests", 400)),
        rate_limit_seconds=float(limits_cfg.get("rate_limit_seconds", 1.0)),
    )

    keywords = list(filters_cfg.get("keyword_gate", []))
    allowed_ct = set([c.lower() for c in filters_cfg.get("allowed_content_types", ["text/html", "application/pdf"])])

    municipalities = ds.get("municipalities") or ds.get("communes") or ds.get("kommunen")
    if not isinstance(municipalities, list) or not municipalities:
        raise ValueError("Dataset config must contain a list: municipalities/communes/kommunen")

    out_root_path = Path(out_root)
    out_root_path.mkdir(parents=True, exist_ok=True)

    for m in municipalities:
        if not isinstance(m, dict):
            continue
        muni_id = str(m.get("id") or m.get("name") or "unknown")
        seed_url = m.get("seed_url") or m.get("start_url") or m.get("url")
        if not isinstance(seed_url, str) or not seed_url.strip():
            raise ValueError(f"Municipality {muni_id} missing seed_url/start_url/url")

        allowed_domains = m.get("allowed_domains") or m.get("domains") or []
        if isinstance(allowed_domains, str):
            allowed_domains = [allowed_domains]
        if not isinstance(allowed_domains, list):
            allowed_domains = []

        crawl_one(
            muni_id=muni_id,
            seed_url=seed_url,
            allowed_domains=[str(d) for d in allowed_domains],
            keywords=keywords,
            allowed_content_types=allowed_ct,
            limits=limits,
            out_dir=out_root_path,  # Path, not str
        )


def crawl_one(
    *,
    muni_id: str,
    seed_url: str,
    allowed_domains: List[str],
    keywords: List[str],
    allowed_content_types: set[str],
    limits: MVPLimits,
    out_dir: Path,
) -> None:
    muni_dir = out_dir / muni_id
    raw_dir = muni_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    index_path = muni_dir / "index.ndjson"

    visited: set[str] = set()
    q: deque[Tuple[str, int, str]] = deque([(seed_url, 0, "seed")])

    docs = 0
    pdfs = 0
    reqs = 0

    while q:
        url, depth, anchor = q.popleft()
        if url in visited:
            continue
        visited.add(url)

        if depth > limits.max_depth:
            continue
        if docs >= limits.max_docs_per_municipality:
            break
        if reqs >= limits.max_total_requests:
            break
        if not _allowed_domain(url, allowed_domains):
            continue

        # seed always; others gated
        if depth > 0 and not (_kw_gate(url, keywords) or _kw_gate(anchor, keywords)):
            continue

        time.sleep(limits.rate_limit_seconds)

        try:
            resp = requests.get(url, timeout=30, headers={"User-Agent": "k3-mvp/1.0"})
            reqs += 1
        except Exception as e:
            _append_ndjson(index_path, {"muni": muni_id, "url": url, "depth": depth, "status": "error", "error": str(e)})
            continue

        ct = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if resp.status_code >= 400:
            _append_ndjson(index_path, {"muni": muni_id, "url": url, "depth": depth, "status": resp.status_code, "ct": ct, "kept": False})
            continue

        if ct not in allowed_content_types:
            _append_ndjson(index_path, {"muni": muni_id, "url": url, "depth": depth, "status": resp.status_code, "ct": ct, "kept": False, "reason": "ct_filtered"})
            continue

        is_pdf = ("pdf" in ct) or url.lower().endswith(".pdf")
        if is_pdf and pdfs >= limits.max_pdfs_per_municipality:
            _append_ndjson(index_path, {"muni": muni_id, "url": url, "depth": depth, "status": resp.status_code, "ct": ct, "kept": False, "reason": "pdf_cap"})
            continue

        content = resp.content
        import hashlib
        digest = hashlib.sha256(content).hexdigest()
        ext = ".pdf" if is_pdf else ".html"
        fp = raw_dir / f"{digest}{ext}"
        fp.write_bytes(content)

        docs += 1
        if is_pdf:
            pdfs += 1

        _append_ndjson(index_path, {"muni": muni_id, "url": url, "depth": depth, "status": resp.status_code, "ct": ct, "kept": True, "raw_path": str(fp)})

        # only enqueue links from HTML
        if not is_pdf and depth < limits.max_depth:
            for nxt, txt in _extract_links_html(url, content):
                if not _allowed_domain(nxt, allowed_domains):
                    continue
                if _kw_gate(nxt, keywords) or _kw_gate(txt, keywords):
                    q.append((nxt, depth + 1, txt))


def _append_ndjson(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
