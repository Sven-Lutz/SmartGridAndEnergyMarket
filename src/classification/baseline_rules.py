# src/classification/baseline_rules.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .domain_taxonomy import (
    POLICY_AREAS,
    INSTRUMENT_TYPES,
    TARGET_SECTORS,
    CLIMATE_DIMENSIONS,
    GOVERNANCE_TYPES,
)

logger = logging.getLogger(__name__)


@dataclass
class BaselineInput:
    candidate_id: str
    municipality_id: Optional[str]
    document_id: Optional[str]
    title: str
    text: str
    url: Optional[str]


# -------------------------------------------------
# kleine Text-Helper
# -------------------------------------------------


def _norm(text: str) -> str:
    return text.lower()


def _any_match(text: str, patterns: List[str], collected: List[str]) -> bool:
    """
    Prüft, ob einer der Strings in patterns im (kleingeschriebenen) Text vorkommt.
    Getroffene Patterns werden in collected eingetragen (für Debug / Confidence).
    """
    t = _norm(text)
    hit = False
    for p in patterns:
        if p in t:
            collected.append(p)
            hit = True
    return hit


# -------------------------------------------------
# Heuristiken pro Label
# -------------------------------------------------


def _guess_policy_area(text: str, kw: List[str]) -> str:
    if _any_match(text, ["klimaschutzkonzept", "klimaplan", "klimaaktionsplan"], kw):
        return "cross_sectoral"
    if _any_match(text, ["gebaeude", "gebäude", "heizung", "sanierung"], kw):
        return "buildings"
    if _any_match(text, ["strom", "waerme", "wärme", "energieversorgung", "fernwärme", "fernwaerme"], kw):
        return "energy_supply"
    if _any_match(text, ["verkehr", "mobilitaet", "mobilität", "radverkehr", "fahrrad"], kw):
        return "mobility"
    if _any_match(text, ["anpassung", "hitzeaktionsplan", "hochwasser", "klimaanpassung"], kw):
        return "adaptation"
    if _any_match(text, ["abfall", "muell", "müll", "recycling"], kw):
        return "waste"
    return "other"


def _guess_instrument_type(text: str, kw: List[str]) -> str:
    if _any_match(text, ["klimaschutzkonzept", "klimaplan", "leitbild", "strategie"], kw):
        return "strategy_plan"
    if _any_match(text, ["satzung", "pflicht", "verordnung", "bauordnung"], kw):
        return "regulation_standard"
    if _any_match(text, ["bau", "umbau", "infrastruktur", "radweg", "ladestation"], kw):
        return "investment_infrastructure"
    if _any_match(text, ["foerderprogramm", "förderprogramm", "zuschuss", "förderung"], kw):
        return "subsidy_funding_program"
    if _any_match(text, ["beratung", "kampagne", "informationsangebot", "infoveranstaltung"], kw):
        return "information_advice"
    if _any_match(text, ["vergab", "beschaffung", "ausschreibung"], kw):
        return "public_procurement"
    return "other"


def _guess_target_sectors(text: str, kw: List[str]) -> List[str]:
    sectors: List[str] = []

    if _any_match(
        text,
        ["rathaus", "staedtische gebaeude", "städtische gebäude", "kommunale gebäude"],
        kw,
    ):
        sectors.append("municipal_buildings")
    if _any_match(text, ["wohngebaeude", "wohngebäude", "eigenheim", "mietshaus"], kw):
        sectors.append("households")  # private Wohngebäude → Haushalte
    if _any_match(text, ["verkehr", "auto", "radverkehr", "oepnv", "öpnv", "bus", "bahn"], kw):
        sectors.append("transport")
    if _any_match(text, ["industrie", "gewerbegebiet", "gewerbe"], kw):
        sectors.append("industry")
    if _any_match(text, ["haushalt", "privathaushalt", "bürger", "bürgerinnen"], kw):
        sectors.append("households")

    if not sectors:
        # viele kommunale Dokumente sind querschnittlich
        sectors.append("cross_sectoral")
    return sectors


def _guess_climate_dimension(text: str, kw: List[str]) -> str:
    mit = _any_match(text, ["co2", "treibhausgas", "emission", "einsparung", "klimaschutz"], kw)
    ada = _any_match(text, ["anpassung", "hitze", "hochwasser", "regenwassermanagement", "resilienz"], kw)

    if mit and ada:
        return "both"
    if mit:
        return "mitigation"
    if ada:
        return "adaptation"
    return "unspecified"


def _guess_governance_type(municipality_id: Optional[str]) -> str:
    # aktuell: alles lokal (Paper-Fokus)
    if municipality_id:
        return "local_municipality"
    return "local_municipality"


def _guess_funding_available(text: str, kw: List[str]) -> bool:
    return _any_match(
        text,
        ["foerderung", "förderung", "zuschuss", "förderprogramm", "foerderprogramm"],
        kw,
    )


# -------------------------------------------------
# Hauptfunktionen
# -------------------------------------------------


def classify_policy_candidate(
    candidate: Dict[str, Any],
    scope_name: str | None = None,
) -> Dict[str, Any]:
    """
    Regelbasierte Baseline-Klassifikation für einen policy_candidate.

    Erwartete Eingabefelder (falls vorhanden):
      - candidate_id
      - municipality_id / municipality
      - document_id
      - title
      - snippet / text
      - url / source_url
    """
    cid = candidate.get("candidate_id") or candidate.get("id") or ""
    muni = candidate.get("municipality_id") or candidate.get("municipality")
    doc_id = candidate.get("document_id")
    title = candidate.get("title") or ""
    text = candidate.get("snippet") or candidate.get("text") or ""
    url = candidate.get("url") or candidate.get("source_url")

    bi = BaselineInput(
        candidate_id=str(cid),
        municipality_id=muni,
        document_id=doc_id,
        title=title,
        text=text,
        url=url,
    )

    full_text = (bi.title + "\n" + bi.text).strip()
    if not full_text:
        full_text = ""

    triggered_keywords: List[str] = []

    policy_area = _guess_policy_area(full_text, triggered_keywords)
    instrument_type = _guess_instrument_type(full_text, triggered_keywords)
    target_sectors = _guess_target_sectors(full_text, triggered_keywords)
    climate_dimension = _guess_climate_dimension(full_text, triggered_keywords)
    governance_type = _guess_governance_type(bi.municipality_id)
    funding_available = _guess_funding_available(full_text, triggered_keywords)

    # -------------------------------------------------
    # Normalisierung auf Domain-Taxonomie
    # -------------------------------------------------

    if policy_area not in POLICY_AREAS:
        policy_area = "other"

    if instrument_type not in INSTRUMENT_TYPES:
        instrument_type = "other"

    target_sectors = [s for s in target_sectors if s in TARGET_SECTORS]
    if not target_sectors:
        # fallback: querschnittlich
        if "cross_sectoral" in TARGET_SECTORS:
            target_sectors = ["cross_sectoral"]
        else:
            target_sectors = ["other"]

    if climate_dimension not in CLIMATE_DIMENSIONS:
        climate_dimension = "unspecified"

    if governance_type not in GOVERNANCE_TYPES:
        governance_type = "local_municipality"

    # einfache Confidence-Heuristik: je mehr Keywords, desto höher
    confidence = min(1.0, 0.2 + 0.1 * len(set(triggered_keywords)))

    measure_title = bi.title or (full_text[:120] + ("..." if len(full_text) > 120 else ""))

    result: Dict[str, Any] = {
        "candidate_id": bi.candidate_id,
        "municipality_id": bi.municipality_id,
        "municipality": candidate.get("municipality"),
        "document_id": bi.document_id,
        "source_document_id": bi.document_id,
        "source_url": bi.url,
        "measure_title": measure_title,
        "policy_area": policy_area,
        "instrument_type": instrument_type,
        "target_sectors": target_sectors,
        "climate_dimension": climate_dimension,
        "governance_type": governance_type,
        "funding_available": funding_available,
        "baseline_keywords": sorted(set(triggered_keywords)),
        "keyword_hit": bool(triggered_keywords),
        "label_source": "baseline_rules_v1",
        "confidence_score": confidence,
    }

    return result


def classify_policy_candidates_batch(
    candidates: Iterable[Dict[str, Any]],
    scope_name: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Batch-Wrapper für die CLI.

    :param candidates: Iterator über policy_candidate-Dictionaries
    :param scope_name: aktuell nur fürs Logging genutzt
    """
    candidate_list = list(candidates)

    measures: List[Dict[str, Any]] = []
    for cand in candidate_list:
        try:
            m = classify_policy_candidate(cand, scope_name=scope_name)
            measures.append(m)
        except Exception as exc:
            logger.warning(
                "Fehler in baseline-Klassifikation für candidate_id=%s: %s",
                cand.get("candidate_id") or cand.get("id"),
                exc,
            )

    logger.info(
        "Baseline-Klassifikation (%s): %d Kandidaten -> %d Maßnahmen",
        scope_name or "-",
        len(candidate_list),
        len(measures),
    )
    return measures
