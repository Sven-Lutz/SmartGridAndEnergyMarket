# src/extraction/__init__.py
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from typing import Any, Dict, Iterable

from bs4 import BeautifulSoup

from src.utils.ndjson import iter_ndjson  # neu

# Projekt-Root ermitteln, analog zu deinen Scripts
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.pdf_extractor import extract_pdf_text_from_url  # type: ignore[import]

logger = logging.getLogger(__name__)

def _extract_from_html(html: str) -> tuple[str, str]:
    """
    Extrahiert (title, text) aus HTML.

    Gibt (title, text) zurück, wobei title leer sein kann.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Titel: <title> oder h1 als Fallback
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    else:
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            title = h1.get_text(strip=True)

    # Volltext: einfacher get_text, später ggf. verfeinern
    text = soup.get_text(separator=" ", strip=True)
    return title, text


def run_extraction(
    scope: str,
    data_root: Path | None = None,
    dry_run: bool = False,
    force: bool = False,
    scope_config: Dict[str, Any] | None = None,  # <--- NEU: wird aktuell nicht genutzt
) -> None:
    """
    Führt die Extraction für einen Scope aus.

    Erwartet:
        - Input:  data/<scope>/staging/documents_raw.ndjson
        - Output: data/<scope>/extracted/documents_extracted.ndjson

    Die Rohdokumente stammen aus dem Crawler (RawDocument-Struktur).
    """

    if data_root is None:
        data_root = PROJECT_ROOT / "data"
    data_root = data_root.resolve()

    # Wenn data_root bereits der Scope-Ordner ist (z.B. .../data/de_by),
    # hänge den Scope nicht noch einmal an.
    if data_root.name == scope:
        scope_root = data_root
    else:
        scope_root = data_root / scope

    input_path = scope_root / "staging" / "documents_raw.ndjson"
    output_dir = scope_root / "extracted"
    output_path = output_dir / "documents_extracted.ndjson"


    logger.info(
        "run_extraction gestartet für scope='%s', dry_run=%s, force=%s",
        scope,
        dry_run,
        force,
    )
    logger.info("Input : %s", input_path)
    logger.info("Output: %s", output_path)

    # ab hier kannst du den bestehenden Body unverändert lassen ...
    # (HTML/PDF-Erkennung, Schleife, Logging etc.)


    if not input_path.exists():
        raise FileNotFoundError(f"Eingabedatei nicht gefunden: {input_path}")

    if output_path.exists() and not force and not dry_run:
        logger.info(
            "Output-Datei existiert bereits und force=False. Extraction wird übersprungen."
        )
        return

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_f = output_path.open("w", encoding="utf-8")
    else:
        out_f = None

    total_in = 0
    total_out = 0
    total_html = 0
    total_pdf = 0
    total_skipped_empty = 0

    try:
        for raw_doc in iter_ndjson(input_path):
            total_in += 1

            # MIME-Typ robuster bestimmen (Crawler-Feld kann unterschiedlich heißen)
            mime_type_raw = (
                raw_doc.get("mime_type")
                or raw_doc.get("content_type")
                or raw_doc.get("headers", {}).get("Content-Type")
                or ""
            )
            mime_type = str(mime_type_raw).lower()

            url = raw_doc.get("source_url") or raw_doc.get("url") or ""
            url_lower = url.lower()

            html = raw_doc.get("raw_content") or ""

            municipality_id = raw_doc.get("municipality_id")
            document_id = raw_doc.get("document_id")

            title = ""
            text = ""

            # Erkennen, ob es sich um ein PDF handelt:
            is_pdf = ("pdf" in mime_type) or url_lower.split("?", 1)[0].endswith(".pdf")

            # 1) HTML-Fall
            if (mime_type.startswith("text/html") or (not is_pdf and html)) and html:
                total_html += 1
                title, text = _extract_from_html(html)

            # 2) PDF-Fall (Content-Type application/pdf ODER URL-Endung .pdf)
            elif is_pdf:
                total_pdf += 1
                # Falls mime_type leer ist, setzen wir explizit application/pdf
                if not mime_type:
                    mime_type = "application/pdf"

                logger.debug(
                    "Beginne PDF-Extraktion (municipality=%s, document_id=%s, url=%s)",
                    municipality_id,
                    document_id,
                    url,
                )
                try:
                    # Begrenzung z. B. auf 40 Seiten
                    text = extract_pdf_text_from_url(url, max_pages=40)
                    # Fallback-Titel: vorhandener Titel oder Dateiname
                    title = raw_doc.get("title") or url.split("/")[-1]
                except Exception as exc:  # pragma: no cover
                    logger.warning(
                        "PDF-Extraktion fehlgeschlagen (URL=%s, municipality=%s, document_id=%s): %s",
                        url,
                        municipality_id,
                        document_id,
                        exc,
                    )
                    text = ""
                    title = raw_doc.get("title") or url.split("/")[-1]

            # 3) Sonstige Typen ignorieren (Bilder, JS, CSS, ...)
            else:
                # Für Debugging interessant:
                logger.debug(
                    "Ignoriere Dokument (weder HTML noch PDF) mime_type=%r url=%r",
                    mime_type_raw,
                    url,
                )
                continue

            # Leere Texte optional überspringen:
            # Für HTML wollen wir i.d.R. nur Einträge mit Inhalt.
            # Für PDFs kann es gescannte Dokumente geben -> wir behalten sie,
            # auch wenn text leer ist, damit wir wenigstens URL/Meta haben.
            if not text.strip():
                if not is_pdf:
                    total_skipped_empty += 1
                    logger.debug(
                        "Überspringe leeres Nicht-PDF-Dokument (municipality=%s, document_id=%s, url=%s)",
                        municipality_id,
                        document_id,
                        url,
                    )
                    continue
                else:
                    logger.info(
                        "PDF ohne extrahierbaren Text behalten (vermutlich gescannt) "
                        "(municipality=%s, document_id=%s, url=%s)",
                        municipality_id,
                        document_id,
                        url,
                    )

            extracted: Dict[str, Any] = {
                "municipality": municipality_id,
                "document_id": document_id,
                "url": url,
                "title": title,
                "text": text,
                "mime_type": mime_type,
                # Meta-Felder übernehmen, falls später gebraucht:
                "referrer_url": raw_doc.get("referrer_url"),
                "http_status": raw_doc.get("http_status"),
                "crawl_timestamp": raw_doc.get("crawl_timestamp"),
                "depth": raw_doc.get("depth"),
            }

            total_out += 1
            if not dry_run and out_f is not None:
                out_f.write(json.dumps(extracted, ensure_ascii=False))
                out_f.write("\n")
    finally:
        if out_f is not None:
            out_f.close()

    logger.info(
        "run_extraction beendet für scope='%s'. %d Rohdokumente gelesen, %d Einträge nach %s geschrieben.",
        scope,
        total_in,
        total_out,
        output_path,
    )
    logger.info(
        "Statistik: HTML=%d, PDF=%d, übersprungen_leer(nicht-PDF)=%d",
        total_html,
        total_pdf,
        total_skipped_empty,
    )
