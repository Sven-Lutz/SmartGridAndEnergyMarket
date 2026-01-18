from __future__ import annotations

from typing import Any, Dict
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


def parse_html_document(html: str, base_url: str) -> Dict[str, Any]:
    """
    Parsed ein HTML-Dokument und extrahiert:
      - title
      - clean_text
      - has_form
      - has_pdf_download
      - has_contact_email
    """
    soup = BeautifulSoup(html, "html.parser")

    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    else:
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            title = h1.get_text(strip=True)

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    clean_text = "\n".join(line for line in text.splitlines() if line.strip())

    has_form = bool(soup.find("form"))

    has_pdf_download = False
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = urljoin(base_url, href)
        if full_url.lower().endswith(".pdf"):
            has_pdf_download = True
            break

    has_contact_email = "mailto:" in html

    return {
        "title": title,
        "clean_text": clean_text,
        "has_form": has_form,
        "has_pdf_download": has_pdf_download,
        "has_contact_email": has_contact_email,
    }
