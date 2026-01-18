from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, Optional

from pypdf import PdfReader

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


def parse_pdf_bytes(pdf_bytes: bytes, max_pages: int = 10) -> Dict[str, Any]:
    """
    Extrahiert Text + Titel aus einem PDF-Byte-Stream (MVP).
    - Kein OCR
    - Keine Layout-/Tabellenlogik
    - max_pages cap für Kosten/Robustheit
    """
    reader = None
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception as e:
        logger.debug("PDF konnte nicht gelesen werden: %s", e)
        return {"title": None, "clean_text": ""}

    # Titel aus Metadaten (optional)
    title: Optional[str] = None
    try:
        meta = reader.metadata
        if meta and getattr(meta, "title", None):
            t = meta.title
            if isinstance(t, str) and t.strip():
                title = t.strip()
    except Exception:
        pass

    texts: list[str] = []
    for i, page in enumerate(reader.pages):
        if max_pages is not None and i >= max_pages:
            break
        try:
            page_text = page.extract_text() or ""
            if page_text.strip():
                texts.append(page_text)
        except Exception as e:
            logger.debug("Fehler beim Text-Extrakt aus PDF-Seite: %s", e)

    clean_text = "\n".join(t.strip() for t in texts if t.strip()).strip()
    return {"title": title, "clean_text": clean_text}
