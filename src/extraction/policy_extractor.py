# src/extraction/policy_extractor.py
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from ..utils.logging_config import get_logger
from ..utils.ndjson_io import read_ndjson

logger = get_logger(__name__)

# -------------------------------------------------
# Heuristik-Parameter
# -------------------------------------------------

# Mindest- und Maximallänge für Snippets (Zeichen)
MIN_SNIPPET_CHARS = 150
MAX_SNIPPET_CHARS = 900

# Positive Keywords: Hinweise auf konkrete Maßnahmen / Programme
POSITIVE_KEYWORDS = [
    # Förder-/Finanzierungsbegriffe
    "förderprogramm",
    "förderung",
    "zuschuss",
    "zuschüsse",
    "zuwendung",
    "fördermittel",
    "förderantrag",
    "förderfäh",
    "förderfähig",
    "bezuschusst",
    "fördertopf",
    "zuwendungsrichtlinie",
    "förderbedingungen",

    # Programme / Konzepte / Pläne
    "programm",
    "aktionsprogramm",
    "maßnahme",
    "maßnahmenpaket",
    "maßnahmenprogramm",
    "klimaschutzkonzept",
    "energie- und klimaschutzkonzept",
    "informationskampagne",
    "kampagne",
    "aktionsplan",
    "aktionsplan klima",
    "strategie",
    "strategiekonzept",
    "kommunale wärmeplanung",
    "wärmeplanung",
    "solaroffensive",
    "solarinitiative",
    "sanierungsprogramm",
    "sanierungsförderung",
    "energieberatung",
    "klimaberatung",
    "beratung und förderung",

    # Typische Satzanfänge in Programmbeschreibungen
    "die stadt fördert",
    "die stadt unterstützt",
    "es wird gefördert",
    "förderfähig sind",
    "sie können einen antrag stellen",
    "anträge können gestellt werden",
    "es besteht die möglichkeit",
]

# Negative Keywords: Navigations-/Meta-/Portalelemente, Newslisten
NEGATIVE_KEYWORDS = [
    # generische Navigations-/Meta-Begriffe
    "sitemap",
    "startseite",
    "impressum",
    "datenschutz",
    "cookie",
    "barrierefreiheit",
    "navigation",
    "breadcrumb",
    "hauptmenü",
    "seiteninhalte",
    "zur übersicht",
    "zurück zur startseite",
    "kontaktformular",
    "copyright",
    "bildnachweis",
    "direkt zu",
    "alle filter zurücksetzen",
    "menü stadt & rathaus",
    "menü eservice",
    "hauptregion der seite anspringen",

    # typische Portalelemente der Kommunen
    "eservice – ihr anliegen bequem online erledigen",
    "virtuelles amt",
    "online-zulassung über i-kfz",
    "meldebescheinigung",
    "wohnsitz elektronisch anmelden",

    # Datums-/Newslisten: Monate + Jahrkombis → meist Veranstaltungskalender
    "januar 20",
    "februar 20",
    "märz 20",
    "april 20",
    "mai 20",
    "juni 20",
    "juli 20",
    "august 20",
    "september 20",
    "oktober 20",
    "november 20",
    "dezember 20",
]

# -------------------------------------------------
# KeywordMatcher für rules/keywords.yml
# -------------------------------------------------


class KeywordMatcher:
    def __init__(self, cfg: dict[str, Any]) -> None:
        general = cfg.get("general", {})
        thematic = cfg.get("thematic", {})
        matching = cfg.get("matching", {})

        self.case_insensitive: bool = bool(matching.get("case_insensitive", True))
        self.min_general: int = int(matching.get("minimum_general_positive", 1))
        self.min_thematic: int = int(matching.get("minimum_thematic_positive", 1))

        self.general_pos = set(general.get("positive", []))
        self.general_neg = set(general.get("negative", []))

        self.thematic_pos: dict[str, set[str]] = {
            k: set(v.get("include", [])) for k, v in thematic.items()
        }
        self.thematic_neg: dict[str, set[str]] = {
            k: set(v.get("exclude", [])) for k, v in thematic.items()
        }

        # flache Menge aller thematischen Include-Keywords
        self._all_thematic_pos = set().union(*self.thematic_pos.values()) if self.thematic_pos else set()

    def _norm(self, text: str) -> str:
        return text.lower() if self.case_insensitive else text

    def score_text(self, text: str) -> dict[str, Any]:
        t = self._norm(text)

        # general
        gen_hits = [kw for kw in self.general_pos if kw in t]
        gen_neg_hits = [kw for kw in self.general_neg if kw in t]

        # thematische Hits
        thematic_hits: dict[str, List[str]] = {}
        for group, kws in self.thematic_pos.items():
            hits = [kw for kw in kws if kw in t]
            if hits:
                thematic_hits[group] = hits

        # thematische Excludes
        thematic_neg_hits: dict[str, List[str]] = {}
        for group, kws in self.thematic_neg.items():
            hits = [kw for kw in kws if kw in t]
            if hits:
                thematic_neg_hits[group] = hits

        has_general = len(gen_hits) >= self.min_general
        has_thematic = len(thematic_hits) >= self.min_thematic
        has_negative = bool(gen_neg_hits) or bool(thematic_neg_hits)

        passes = has_general and has_thematic and not has_negative

        # einfache Confidence-Heuristik
        n_pos = len(gen_hits) + sum(len(v) for v in thematic_hits.values())
        confidence = min(1.0, 0.1 + 0.05 * n_pos)

        return {
            "passes_filter": passes,
            "general_hits": gen_hits,
            "thematic_hits": thematic_hits,
            "negative_hits": {
                "general": gen_neg_hits,
                "thematic": thematic_neg_hits,
            },
            "confidence": confidence,
        }


def load_keyword_matcher(keyword_config_path: Path) -> KeywordMatcher:
    with keyword_config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return KeywordMatcher(cfg)


# -------------------------------------------------
# Hilfsfunktionen
# -------------------------------------------------


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_into_paragraphs(text: str) -> List[str]:
    """
    Teilt Text in Absätze auf Basis von Leerzeilen.
    Falls keine Leerzeilen vorkommen, wird ein einzelner Absatz zurückgegeben.
    """
    if not text:
        return []

    # Erst in Roh-Absätze aufspalten
    raw_paragraphs = re.split(r"\n\s*\n", text)
    paragraphs: List[str] = []

    for p in raw_paragraphs:
        p = _normalize_whitespace(p)
        if not p:
            continue
        paragraphs.append(p)

    return paragraphs


def _chunk_long_paragraph(paragraph: str, max_len: int) -> Iterable[str]:
    """
    Zerschneidet sehr lange Absätze in kleinere Snippets von max_len Zeichen.
    Versucht, an Satzgrenzen (Punkt) zu schneiden, fällt sonst auf harte Cuts zurück.
    """
    text = paragraph.strip()
    if len(text) <= max_len:
        yield text
        return

    # Erst grob in Sätze splitten
    sentences = re.split(r"(?<=[.!?])\s+", text)
    current = ""

    for s in sentences:
        if not s:
            continue
        if not current:
            current = s
            continue

        # Wenn der nächste Satz noch in das aktuelle Fenster passt
        if len(current) + 1 + len(s) <= max_len:
            current = current + " " + s
        else:
            # aktuellen Block yielden und neu anfangen
            yield current.strip()
            current = s

    if current:
        yield current.strip()


def _iter_snippet_candidates(text: str) -> Iterable[str]:
    """
    Erzeugt potenzielle Snippets aus einem Dokument:
    - Absatzweise Splits
    - Zu lange Absätze werden in kleinere Blöcke gechunkt.
    """
    for para in _split_into_paragraphs(text):
        if len(para) <= MAX_SNIPPET_CHARS:
            yield para
        else:
            for chunk in _chunk_long_paragraph(para, MAX_SNIPPET_CHARS):
                yield chunk


def _is_snippet_candidate(
    snippet: str,
    matcher: KeywordMatcher | None = None,
) -> Tuple[bool, bool, Dict[str, Any] | None]:
    """
    Prüft, ob ein Snippet als Policy-Kandidat taugt.

    Rückgabe:
      (ist_kandidat, keyword_hit_simple, matcher_score_dict_or_None)

    - Länge zwischen MIN_SNIPPET_CHARS und MAX_SNIPPET_CHARS
    - enthält mindestens ein POSITIVE_KEYWORD (simple Heuristik)
    - enthält kein NEGATIVE_KEYWORD
    - falls matcher gesetzt: matcher.passes_filter muss True sein
    """
    s_norm = snippet.strip()
    if not s_norm:
        return False, False, None

    if len(s_norm) < MIN_SNIPPET_CHARS:
        return False, False, None

    lower = s_norm.lower()

    if any(bad in lower for bad in NEGATIVE_KEYWORDS):
        return False, False, None

    keyword_hit_simple = any(kw in lower for kw in POSITIVE_KEYWORDS)
    if not keyword_hit_simple and matcher is None:
        # ohne Matcher verlangen wir mindestens ein POSITIVE_KEYWORD
        return False, False, None

    score: Dict[str, Any] | None = None
    if matcher is not None:
        score = matcher.score_text(s_norm)
        if not score.get("passes_filter", False):
            return False, keyword_hit_simple, score

    return True, keyword_hit_simple, score


# -------------------------------------------------
# Hauptfunktion: build_policy_candidates
# -------------------------------------------------


def build_policy_candidates(
    *,
    extracted_path: Path,
    out_path: Path,
    keyword_config_path: Path | None = None,
) -> int:
    """
    Erzeugt policy_candidates.ndjson aus documents_extracted.ndjson
    mithilfe einer einfachen Absatz-/Keyword-Heuristik + rules/keywords.yml (falls vorhanden).

    Erwartetes Input-Schema (pro Dokument, verkürzt):
      {
        "municipality_id": "...",
        "municipality": "...",
        "document_id": "muenchen::0",
        "url": "...",
        "title": "...",
        "text": "..."
      }

    Output-Schema (pro Candidate, Beispielschema):
      {
        "candidate_id": "muenchen::0::p0",
        "document_id": "muenchen::0",
        "municipality_id": "...",
        "municipality": "...",
        "title": "...",
        "snippet": "...",
        "url": "...",
        "policy_area": null,
        "instrument_type": null,
        "target_sectors": null,
        "climate_dimension": null,
        "governance_type": null,
        "funding_available": null,
        "keyword_hit": true,
        "confidence_score": 0.0,
        "keyword_hits_general": [...],
        "keyword_hits_thematic": {...},
        "keyword_negative_hits": {...}
      }
    """

    if not extracted_path.exists():
        raise FileNotFoundError(extracted_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Baue policy_candidates aus: %s", extracted_path)
    logger.info("Schreibe nach: %s", out_path)

    # KeywordMatcher laden (falls konfiguriert)
    matcher: KeywordMatcher | None = None
    if keyword_config_path is not None:
        if keyword_config_path.exists():
            try:
                matcher = load_keyword_matcher(keyword_config_path)
                logger.info("Keyword-Konfiguration aus %s geladen.", keyword_config_path)
            except Exception as exc:
                logger.warning(
                    "Konnte Keyword-Konfiguration %s nicht laden (%s) – verwende nur POSITIVE/NEGATIVE-Heuristik.",
                    keyword_config_path,
                    exc,
                )
                matcher = None
        else:
            logger.warning(
                "Keyword-Konfiguration %s existiert nicht – verwende nur POSITIVE/NEGATIVE-Heuristik.",
                keyword_config_path,
            )

    num_docs = 0    # Anzahl gelesener Dokumente
    num_candidates = 0  # Anzahl geschriebener Kandidaten

    # Zähler, um pro Dokument eindeutige Candidate-IDs zu erzeugen
    per_doc_counter: Dict[Tuple[str, str], int] = defaultdict(int)

    with out_path.open("w", encoding="utf-8") as fout:
        for doc in read_ndjson(extracted_path):
            num_docs += 1

            municipality_id = doc.get("municipality_id")
            municipality = doc.get("municipality")
            document_id = doc.get("document_id") or doc.get("id") or f"doc{num_docs}"
            title = doc.get("title") or ""
            url = doc.get("url") or doc.get("source_url") or None
            text = doc.get("text") or ""

            if not text:
                continue

            key = (municipality_id or "", document_id)
            local_idx = per_doc_counter[key]

            for snippet in _iter_snippet_candidates(text):
                ok, keyword_hit, score = _is_snippet_candidate(snippet, matcher=matcher)
                if not ok:
                    continue

                candidate_id = f"{municipality_id or 'unknown'}::{document_id}::p{local_idx}"
                local_idx += 1

                candidate: Dict[str, Any] = {
                    "candidate_id": candidate_id,
                    "document_id": document_id,
                    "municipality_id": municipality_id,
                    "municipality": municipality,
                    "title": title,
                    "snippet": snippet,
                    "url": url,
                    # Labels werden später per Baseline/LLM vergeben
                    "policy_area": None,
                    "instrument_type": None,
                    "target_sectors": None,
                    "climate_dimension": None,
                    "governance_type": None,
                    "funding_available": None,
                    "keyword_hit": keyword_hit,
                }

                # KeywordMatcher-Infos ergänzen (falls vorhanden)
                if score is not None:
                    candidate["confidence_score"] = score.get("confidence", 0.0)
                    candidate["keyword_hits_general"] = score.get("general_hits", [])
                    candidate["keyword_hits_thematic"] = score.get("thematic_hits", {})
                    candidate["keyword_negative_hits"] = score.get("negative_hits", {})
                else:
                    candidate["confidence_score"] = 0.0
                    candidate["keyword_hits_general"] = []
                    candidate["keyword_hits_thematic"] = {}
                    candidate["keyword_negative_hits"] = {}

                fout.write(json.dumps(candidate, ensure_ascii=False))
                fout.write("\n")
                num_candidates += 1

            per_doc_counter[key] = local_idx

    logger.info(
        "Policy-Filter abgeschlossen: %d Dokumente gelesen, %d Candidates geschrieben.",
        num_docs,
        num_candidates,
    )

    return num_candidates
