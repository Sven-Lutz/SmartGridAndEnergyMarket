# src/crawl/seed_discovery.py

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urljoin, urlparse

import requests

from .municipal_loader import load_municipalities

logger = logging.getLogger(__name__)


# Basis-Keywordliste für Klima / Energie / Förderung
DEFAULT_KEYWORDS = [
    "klima",
    "klimaschutz",
    "klimaneutral",
    "klimaneutralität",
    "energie",
    "strom",
    "wärme",
    "heizung",
    "heizungs",
    "sanierung",
    "umwelt",
    "nachhalt",
    "mobilität",
    "e-mobilität",
    "emobilität",
    "solar",
    "photovoltaik",
    "pv",
    "wind",
    "fernwärme",
    "foerder",
    "förder",
    "foerderprogramm",
    "förderprogramm",
    "zuschuss",
]

EXCLUDED_EXTENSIONS = [
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".mp4",
    ".mp3",
    ".zip",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
]


@dataclass
class CandidateLink:
    url: str
    anchor_text: str
    score: int


class _LinkExtractor(HTMLParser):
    """Sehr einfacher HTML-Link-Extractor, der href + sichtbaren Text sammelt."""

    def __init__(self) -> None:
        super().__init__()
        self._in_a = False
        self._current_href: str | None = None
        self._current_text_parts: List[str] = []
        self.links: List[Tuple[str, str]] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            self._in_a = True
            self._current_href = None
            self._current_text_parts = []
            for k, v in attrs:
                if k.lower() == "href":
                    self._current_href = v

    def handle_data(self, data):
        if self._in_a and data:
            self._current_text_parts.append(data.strip())

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self._in_a:
            text = " ".join(t for t in self._current_text_parts if t)
            if self._current_href:
                self.links.append((self._current_href, text))
            self._in_a = False
            self._current_href = None
            self._current_text_parts = []


def _is_internal(full_url: str, base_netloc: str) -> bool:
    parsed = urlparse(full_url)
    return parsed.netloc == "" or parsed.netloc == base_netloc


def _has_excluded_extension(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    return any(path.endswith(ext) for ext in EXCLUDED_EXTENSIONS)


def _score_link(full_url: str, anchor_text: str, keywords: List[str]) -> int:
    """Einfache Heuristik: Keywords in Pfad und Linktext zählen."""
    score = 0
    path = urlparse(full_url).path.lower()
    text = anchor_text.lower()

    for kw in keywords:
        if kw in path:
            score += 2
        if kw in text:
            score += 2

    # Bonus, wenn im Pfad etwas wie „/umwelt-klima/“ vorkommt
    if re.search(r"(umwelt.*klima|klima.*umwelt)", path):
        score += 2

    return score


def discover_seeds_for_municipality(
    municipality_id: str,
    base_url: str,
    max_seeds: int = 5,
    keywords: List[str] | None = None,
    timeout: float = 10.0,
) -> List[CandidateLink]:
    """
    Lade die base_url, extrahiere interne Links und bewerte sie mit einer Klima-/Energie-Heuristik.
    Liefert eine Liste der Top-Kandidaten (inkl. Score und Anchor-Text).
    """
    if keywords is None:
        keywords = DEFAULT_KEYWORDS

    logger.info("Seed-Discovery für %s (base_url=%s)", municipality_id, base_url)

    try:
        resp = requests.get(base_url, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning(
            "Seed-Discovery: Konnte base_url für %s nicht laden (%s): %s",
            municipality_id,
            base_url,
            exc.__class__.__name__,
        )
        return []

    html = resp.text
    extractor = _LinkExtractor()
    extractor.feed(html)

    base_netloc = urlparse(base_url).netloc
    candidates: List[CandidateLink] = []

    for href, text in extractor.links:
        if not href:
            continue

        full_url = urljoin(base_url, href)

        if not _is_internal(full_url, base_netloc):
            continue

        if _has_excluded_extension(full_url):
            continue

        score = _score_link(full_url, text, keywords)
        if score <= 0:
            continue

        candidates.append(CandidateLink(url=full_url, anchor_text=text, score=score))

    # nach Score sortieren, höchste zuerst
    candidates.sort(key=lambda c: c.score, reverse=True)

    # Duplikate (gleiche URL) entfernen, Score der ersten Variante behalten
    unique: dict[str, CandidateLink] = {}
    for c in candidates:
        if c.url not in unique:
            unique[c.url] = c

    deduped = list(unique.values())
    top = deduped[:max_seeds]

    logger.info(
        "Seed-Discovery für %s: %d Kandidaten, Top %d Seeds zurückgegeben",
        municipality_id,
        len(deduped),
        len(top),
    )
    return top


def discover_seeds_for_all_municipalities(
    meta_root: Path,
    max_seeds_per_municipality: int = 5,
    keywords: List[str] | None = None,
    timeout: float = 10.0,
) -> Dict[str, List[CandidateLink]]:
    """
    Lädt alle Kommunen aus meta/municipalities.csv (bzw. .ndjson) und
    führt Seed-Discovery für jede durch.

    Rückgabe:
      dict[municipality_id] -> List[CandidateLink]
    """
    municipalities = load_municipalities(meta_root)
    results: Dict[str, List[CandidateLink]] = {}

    for m in municipalities:
        mid = m.get("municipality_id")
        base_url = m.get("base_url")

        if not mid or not base_url:
            logger.warning("Überspringe Municipality ohne ID/base_url: %r", m)
            continue

        seeds = discover_seeds_for_municipality(
            municipality_id=mid,
            base_url=base_url,
            max_seeds=max_seeds_per_municipality,
            keywords=keywords,
            timeout=timeout,
        )
        results[mid] = seeds

    logger.info(
        "Seed-Discovery gesamt: %d Kommunen, %d mit mindestens einem Seed.",
        len(municipalities),
        sum(1 for v in results.values() if v),
    )
    return results
