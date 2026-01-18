# src/extraction/pdf_extractor.py
from __future__ import annotations

import io
import logging
from typing import Optional

import requests
from pypdf import PdfReader

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "k3-pipeline/0.1 (+contact@example.org)"
DEFAULT_MAX_BYTES = 30 * 1024 * 1024  # 30 MB


def _request_pdf(
    url: str,
    timeout: float,
    max_bytes: int,
    user_agent: str,
) -> bytes:
    """
    Lädt ein PDF von einer URL mit einfachen Schutzmechanismen:
    - setzt User-Agent
    - prüft Content-Length (falls vorhanden)
    - begrenzt die maximal zugelassene Größe
    """
    headers = {
        "User-Agent": user_agent,
    }

    # Erster Versuch: HEAD, um Content-Length zu prüfen
    try:
        head_resp = requests.head(url, timeout=timeout, headers=headers, allow_redirects=True)
    except requests.RequestException as exc:
        logger.warning("HEAD-Request für PDF fehlgeschlagen (URL=%s): %s", url, exc)
        head_resp = None

    if head_resp is not None:
        cl = head_resp.headers.get("Content-Length")
        if cl is not None:
            try:
                cl_int = int(cl)
                if cl_int > max_bytes:
                    logger.warning(
                        "PDF zu groß (Content-Length=%d > %d Bytes), breche ab (URL=%s).",
                        cl_int,
                        max_bytes,
                        url,
                    )
                    raise ValueError("PDF zu groß (Content-Length-Limit überschritten)")
            except ValueError:
                # Falls Content-Length nicht parsebar ist, ignorieren wir das
                pass

    # Dann der eigentliche GET-Request
    resp = requests.get(url, timeout=timeout, headers=headers, stream=True)
    resp.raise_for_status()

    # Content-Type sanity check (optional, aber hilfreich)
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "pdf" not in ctype:
        logger.debug(
            "Content-Type sieht nicht nach PDF aus (Content-Type=%r, URL=%s).",
            ctype,
            url,
        )

    # Inhalt begrenzt einlesen
    content = io.BytesIO()
    total = 0
    chunk_size = 64 * 1024  # 64 KB

    for chunk in resp.iter_content(chunk_size=chunk_size):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            logger.warning(
                "PDF-Download überschreitet max_bytes=%d (URL=%s), breche ab.",
                max_bytes,
                url,
            )
            raise ValueError("PDF zu groß (Download-Limit überschritten)")
        content.write(chunk)

    return content.getvalue()


def extract_pdf_text_from_url(
    url: str,
    timeout: float = 20.0,
    max_pages: Optional[int] = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    user_agent: str = DEFAULT_USER_AGENT,
) -> str:
    """
    Lädt ein PDF von einer URL und extrahiert den Text.

    Args:
        url: PDF-URL.
        timeout: Request-Timeout in Sekunden.
        max_pages: Optional Begrenzung der Seitenanzahl (0-basiert gezählt).
        max_bytes: Maximale erlaubte Größe des PDFs in Bytes.
        user_agent: User-Agent-String für HTTP-Requests.

    Returns:
        Extrahierter Text (zusammenhängend, mit Zeilenumbrüchen).
        Bei Fehlern wird ein leerer String zurückgegeben.
    """
    logger.debug("Starte PDF-Extraktion von URL: %s", url)

    try:
        raw_bytes = _request_pdf(
            url=url,
            timeout=timeout,
            max_bytes=max_bytes,
            user_agent=user_agent,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Fehler beim Laden des PDFs (URL=%s): %s", url, exc)
        return ""

    try:
        bio = io.BytesIO(raw_bytes)
        reader = PdfReader(bio)
    except Exception as exc:  # pragma: no cover
        logger.warning("Fehler beim Initialisieren des PdfReader (URL=%s): %s", url, exc)
        return ""

    texts: list[str] = []
    for i, page in enumerate(reader.pages):
        if max_pages is not None and i >= max_pages:
            logger.debug(
                "max_pages=%s erreicht, weitere Seiten werden nicht gelesen (URL=%s).",
                max_pages,
                url,
            )
            break

        try:
            page_text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Fehler bei der PDF-Extraktion (URL=%s, Seite=%d): %s",
                url,
                i,
                exc,
            )
            continue

        if page_text.strip():
            texts.append(page_text)

    text = "\n\n".join(texts).strip()
    logger.debug(
        "PDF-Extraktion abgeschlossen (URL=%s, Seiten_gelesen=%d, Textlänge=%d).",
        url,
        len(texts),
        len(text),
    )
    return text
