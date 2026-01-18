# src/extraction/document_extractor.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

from bs4 import BeautifulSoup

from src.utils.logging_config import get_logger
from src.extraction.pdf_parser import parse_pdf_bytes

logger = get_logger(__name__)

MAX_PDF_PAGES_MVP = 10  # MVP: pypdf only, no OCR


def _iter_ndjson(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"NDJSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _html_to_title_and_text(html: str | None) -> Tuple[str | None, str | None]:
    if not html:
        return None, None

    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else None

    text = soup.get_text(separator="\n")
    if text is not None:
        text = " ".join(text.split()).strip()

    return (title.strip() if isinstance(title, str) and title.strip() else None), (text or None)


def _pdf_path_to_title_and_text(pdf_path: str | None) -> Tuple[str | None, str | None]:
    if not pdf_path:
        return None, None

    p = Path(pdf_path)
    if not p.exists() or p.suffix.lower() != ".pdf":
        return None, None

    try:
        pdf_bytes = p.read_bytes()
    except Exception as e:
        logger.debug("Konnte PDF nicht lesen (%s): %s", pdf_path, e)
        return None, None

    parsed = parse_pdf_bytes(pdf_bytes, max_pages=MAX_PDF_PAGES_MVP)
    title = parsed.get("title")
    text = parsed.get("clean_text")

    if isinstance(text, str):
        text = " ".join(text.split()).strip() or None
    else:
        text = None

    if isinstance(title, str):
        title = title.strip() or None
    else:
        title = None

    return title, text


def _html_path_to_title_and_text(html_path: str | None) -> Tuple[str | None, str | None]:
    if not html_path:
        return None, None

    p = Path(html_path)
    if not p.exists() or p.suffix.lower() not in (".html", ".htm"):
        return None, None

    try:
        raw = p.read_bytes()
    except Exception as e:
        logger.debug("Konnte HTML nicht lesen (%s): %s", html_path, e)
        return None, None

    # try utf-8 first, fallback replace
    try:
        html = raw.decode("utf-8", errors="replace")
    except Exception:
        html = raw.decode(errors="replace")

    return _html_to_title_and_text(html)


def _transform_raw_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unterstützt zwei Raw-Formate:
    1) "normal": source_url + raw_content / raw_path(.pdf)
    2) "mvp":    url + ct + raw_path(.html/.pdf) + muni
    """
    url = rec.get("source_url") or rec.get("url")
    municipality_id = rec.get("municipality_id") or rec.get("muni")
    mime_type = rec.get("mime_type") or rec.get("ct")
    http_status = rec.get("http_status") or rec.get("status")
    depth = rec.get("depth")

    raw_html_inline = rec.get("raw_content")
    raw_path = rec.get("raw_path")

    title: str | None = None
    text: str | None = None
    doc_kind: str = "unknown"

    # 1) MVP / allgemeiner Fall: raw_path kann .pdf ODER .html sein
    if isinstance(raw_path, str) and raw_path:
        suffix = Path(raw_path).suffix.lower()
        if suffix == ".pdf":
            title, text = _pdf_path_to_title_and_text(raw_path)
            doc_kind = "pdf"
        elif suffix in (".html", ".htm"):
            title, text = _html_path_to_title_and_text(raw_path)
            doc_kind = "html"
        else:
            # unbekannt: fallback auf inline
            title, text = _html_to_title_and_text(raw_html_inline)
            doc_kind = "html" if text else "unknown"
    else:
        # 2) normaler Crawl: HTML inline
        title, text = _html_to_title_and_text(raw_html_inline)
        doc_kind = "html" if text else "unknown"

    out: Dict[str, Any] = {
        "municipality_id": municipality_id,
        "url": url,
        "title": title,
        "text": text,
        "doc_kind": doc_kind,       # html|pdf|unknown
        "mime_type": mime_type,
        "http_status": http_status,
        "depth": depth,
        "raw_path": raw_path,
        "kept": rec.get("kept"),
    }
    return out


def extract_documents(raw_path: Path, out_path: Path) -> int:
    logger.info("Extrahiere Dokumente aus: %s", raw_path)
    logger.info("Schreibe extrahierte Dokumente nach: %s", out_path)

    count_in = 0
    count_out = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f_out:
        for rec in _iter_ndjson(raw_path):
            count_in += 1
            extracted = _transform_raw_record(rec)

            # Minimalbedingungen
            if not extracted.get("url"):
                continue
            if not extracted.get("text") and not extracted.get("title"):
                continue

            json.dump(extracted, f_out, ensure_ascii=False)
            f_out.write("\n")
            count_out += 1

    logger.info(
        "Document-Extraction abgeschlossen: %d Rohdokumente gelesen, %d extrahiert.",
        count_in,
        count_out,
    )
    return count_out
