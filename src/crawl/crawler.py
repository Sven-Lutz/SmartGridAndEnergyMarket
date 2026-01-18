from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class RawDocument:
    municipality_id: str
    document_id: str
    source_url: str
    referrer_url: str | None
    mime_type: str | None
    http_status: int | None
    raw_content: str | None  # HTML only (decoded text)
    raw_path: str | None     # PDF only (path to saved file)
    crawl_timestamp: str
    keyword_hit: bool
    depth: int


def _load_discovered_seeds(meta_dir: Path) -> dict[str, list[str]]:
    """
    Lädt Seeds aus data/<scope>/meta/discovered_seeds.ndjson, falls vorhanden.

    Format pro Zeile:
      {
        "municipality_id": "...",
        "base_url": "...",
        "discovered_seeds": ["https://...", ...],
        "debug": { ... }
      }
    """
    path = meta_dir / "discovered_seeds.ndjson"
    if not path.exists():
        logger.info("_load_discovered_seeds: %s existiert nicht. Keine discovered seeds.", path)
        return {}

    out: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            mid = obj.get("municipality_id")
            seeds = obj.get("discovered_seeds")
            if isinstance(mid, str) and isinstance(seeds, list):
                out[mid] = [s for s in seeds if isinstance(s, str)]
    logger.info("_load_discovered_seeds: %d municipalities mit seeds geladen.", len(out))
    return out


def _fetch_content(url: str, timeout: float = 15.0) -> tuple[int | None, str | None, bytes | None, str | None]:
    """
    Holt Content (HTML oder PDF) von einer URL.

    Rückgabe:
      (http_status, mime_type, content_bytes | None, encoding_guess | None)
    """
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            allow_redirects=True,
            headers={"User-Agent": "k3-mvp-crawler/0.1 (+seminararbeit)"},
        )
    except requests.RequestException as exc:
        logger.warning("Fehler beim Abruf von %s: %s", url, exc)
        return None, None, None, None

    status = resp.status_code
    content_type = resp.headers.get("Content-Type", "")
    mime_type = content_type.split(";")[0].strip().lower() if content_type else None

    if status < 200 or status >= 400:
        return status, mime_type, None, None

    # Robust encoding guess for HTML decoding downstream
    enc = resp.encoding
    if not enc or enc.lower() in ("iso-8859-1", "latin-1"):
        enc = resp.apparent_encoding or "utf-8"

    return status, mime_type, resp.content, enc


def _decode_html(content: bytes, encoding_guess: str | None) -> str:
    enc = encoding_guess or "utf-8"
    try:
        return content.decode(enc, errors="replace")
    except Exception:
        return content.decode("utf-8", errors="replace")


def _is_relevant_path(url: str) -> bool:
    """
    Sehr grobe Heuristik, um offensichtliche Nicht-Inhalte auszusortieren.
    (Im MVP bleibt das bewusst simpel.)
    """
    p = urlparse(url).path.lower()
    # harte Excludes (Assets)
    for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".mp4", ".zip", ".doc", ".docx", ".xls", ".xlsx"):
        if p.endswith(ext):
            return False
    return True


def _extract_links(base_url: str, html: str) -> list[str]:
    """
    Extrahiert Links (a[href]) und normalisiert sie auf absolute URLs.
    """
    soup = BeautifulSoup(html, "html.parser")
    out: list[str] = []

    for a in soup.find_all("a"):
        href = a.get("href")
        if not isinstance(href, str):
            continue

        abs_url = urljoin(base_url, href)
        out.append(abs_url)

    return out



def _crawl_single_municipality(
    municipality_id: str,
    base_url: str,
    seed_urls: list[str],
    max_pages: int,
    max_depth: int,
    raw_dir: Path,
    max_pdfs_per_municipality: int = 8,
) -> list[RawDocument]:
    """
    BFS-Crawler ab einer Seed-Liste.

    MVP: HTML crawlen + Links verfolgen.
         PDFs nur speichern (kein Parsing), keine Link-Extraktion aus PDFs.
    """
    from datetime import datetime

    logger.info(
        "Crawle Kommune '%s' (base_url=%s) mit %d Seed-URLs.",
        municipality_id,
        base_url,
        len(seed_urls),
    )

    visited: set[str] = set()
    queue: list[tuple[str, str | None, int]] = [(u, None, 0) for u in seed_urls]

    results: list[RawDocument] = []
    counter = 0
    pdf_counter = 0

    while queue and counter < max_pages:
        url, referrer, depth = queue.pop(0)

        if url in visited:
            continue
        visited.add(url)

        if depth > max_depth:
            continue

        if not _is_relevant_path(url):
            continue

        status, mime_type, content, enc = _fetch_content(url)
        timestamp = datetime.utcnow().isoformat() + "Z"

        raw_content: str | None = None
        raw_path: str | None = None

        # PDF: nur speichern (kein parsing)
        is_pdf = (mime_type == "application/pdf") or url.lower().endswith(".pdf")
        is_html = (mime_type is not None and "text/html" in mime_type)

        if content and is_pdf:
            if pdf_counter < max_pdfs_per_municipality:
                muni_raw_dir = raw_dir / municipality_id
                muni_raw_dir.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256(content).hexdigest()
                fp = muni_raw_dir / f"{digest}.pdf"
                fp.write_bytes(content)
                raw_path = str(fp)
                pdf_counter += 1
            else:
                # PDF-Cap erreicht: wir loggen trotzdem einen Eintrag, aber speichern nicht.
                raw_path = None

        elif content and is_html:
            raw_content = _decode_html(content, enc)

        # keyword_hit bleibt für spätere Extraction/Filtering-Phase
        keyword_hit = False

        doc = RawDocument(
            municipality_id=municipality_id,
            document_id=f"{municipality_id}::{counter}",
            source_url=url,
            referrer_url=referrer,
            mime_type=mime_type,
            http_status=status,
            raw_content=raw_content,
            raw_path=raw_path,
            crawl_timestamp=timestamp,
            keyword_hit=keyword_hit,
            depth=depth,
        )
        results.append(doc)
        counter += 1

        # Nur HTML weiterverfolgen
        if raw_content and depth < max_depth and status and 200 <= status < 300:
            for nxt in _extract_links(base_url, raw_content):
                if nxt not in visited:
                    queue.append((nxt, url, depth + 1))

    logger.info(
        "_crawl_single_municipality: %d Dokumente für '%s' (base_url=%s) gesammelt. PDFs gespeichert: %d/%d",
        len(results),
        municipality_id,
        base_url,
        pdf_counter,
        max_pdfs_per_municipality,
    )
    return results


def crawl_scope(
    scope: str,
    municipalities: list[dict[str, Any]],
    output_path: Path,
    max_municipalities: int,
    max_pages_per_municipality: int,
    max_depth: int,
    data_root: Path,
    max_pdfs_per_municipality: int = 8,
) -> None:
    """
    Hauptfunktion: crawlt alle Kommunen eines Scopes und schreibt documents_raw.ndjson.

    MVP-Änderung:
      - PDFs werden im Crawl nur gespeichert (raw/<municipality_id>/<sha256>.pdf),
        und im NDJSON nur über raw_path referenziert.
      - HTML wird wie bisher als raw_content (Text) gespeichert.
    """
    meta_dir = data_root / "meta"

    logger.info(
        "crawl_scope(%s): %d Municipalities übergeben.",
        scope,
        len(municipalities),
    )

    # discovered_seeds laden (falls vorhanden)
    discovered = _load_discovered_seeds(meta_dir)

    # PDF raw files go next to the NDJSON output
    raw_dir = output_path.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_docs: list[RawDocument] = []

    for i, m in enumerate(municipalities):
        if i >= max_municipalities:
            logger.info(
                "max_municipalities=%d erreicht, weitere Kommunen werden nicht gecrawlt.",
                max_municipalities,
            )
            break

        municipality_id = m.get("id") or m.get("municipality_id") or m.get("ags") or f"m{i+1}"

        # Seeded MVP: bevorzugt seed_url/start_url aus Config; sonst discovered seeds; sonst base_url
        seed_url = m.get("seed_url") or m.get("start_url")
        base_url = m.get("base_url") or seed_url or ""

        seed_urls: list[str]
        if seed_url:
            seed_urls = [seed_url]
        else:
            seed_urls = discovered.get(municipality_id) or ([base_url] if base_url else [])

        if not seed_urls:
            logger.warning("Keine Seed-URL für municipality_id=%s (übersprungen).", municipality_id)
            continue

        docs = _crawl_single_municipality(
            municipality_id=municipality_id,
            base_url=base_url,
            seed_urls=seed_urls,
            max_pages=max_pages_per_municipality,
            max_depth=max_depth,
            raw_dir=raw_dir,
            max_pdfs_per_municipality=max_pdfs_per_municipality,
        )
        all_docs.extend(docs)

    # NDJSON schreiben
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for d in all_docs:
            f.write(json.dumps(d.__dict__, ensure_ascii=False) + "\n")

    logger.info(
        "crawl_scope: Insgesamt %d Dokumente nach %s geschrieben.",
        len(all_docs),
        output_path,
    )
