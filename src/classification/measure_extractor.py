# src/classification/measure_extractor.py
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Optional

from ..utils.logging_config import get_logger
from .llm_client import LLMClient
from .label_schemas import (
    MeasureLabels,
    PolicyArea,
    InstrumentType,
    TargetSector,
    ClimateDimension,
    GovernanceType,
    FundingAvailable,
)

logger = get_logger(__name__)


# -------------------------------------------------
# Hilfsfunktionen
# -------------------------------------------------


def _match_enum_by_value(enum_cls, raw: Any):
    """
    Versucht, einen Enum-Wert über seinen .value zu finden.
    Gibt None zurück, wenn kein Match gefunden wird.
    """
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None

    for member in enum_cls:
        if text == str(member.value).lower():
            return member
    return None


def _parse_policy_area(raw: Any) -> Optional[PolicyArea]:
    pa = _match_enum_by_value(PolicyArea, raw)
    return pa or getattr(PolicyArea, "OTHER", None)


def _parse_instrument_type(raw: Any) -> Optional[InstrumentType]:
    it = _match_enum_by_value(InstrumentType, raw)
    if it is not None:
        return it

    # Heuristik auf Basis des Texts
    text = (str(raw) if raw is not None else "").lower()
    if not text:
        return getattr(InstrumentType, "OTHER", None)

    if "förder" in text or "zuschuss" in text or "funding" in text or "subsid" in text:
        return getattr(InstrumentType, "SUBSIDY_FUNDING_PROGRAM", None)
    if "strategie" in text or "konzept" in text or "plan" in text:
        return getattr(InstrumentType, "STRATEGY_PLAN", None)
    if "gesetz" in text or "satzung" in text or "verordnung" in text or "regulation" in text:
        return getattr(InstrumentType, "REGULATION_STANDARD", None)
    if "beschaffung" in text or "procurement" in text:
        return getattr(InstrumentType, "PUBLIC_PROCUREMENT", None)
    if "infrastruktur" in text or "infrastructure" in text:
        return getattr(InstrumentType, "INVESTMENT_INFRASTRUCTURE", None)

    return getattr(InstrumentType, "OTHER", None)


def _parse_climate_dimension(raw: Any) -> Optional[ClimateDimension]:
    cd = _match_enum_by_value(ClimateDimension, raw)
    return cd or getattr(ClimateDimension, "UNSPECIFIED", None)


def _parse_governance_type(raw: Any) -> Optional[GovernanceType]:
    gt = _match_enum_by_value(GovernanceType, raw)
    return gt or getattr(GovernanceType, "LOCAL_MUNICIPALITY", None)


def _normalize_sectors_enum(raw_val: Any) -> List[TargetSector]:
    """
    Normalisiert target_sector / target_sectors auf eine Liste von TargetSector-Enums.
    Enthält auch eine einfache Heuristik für freie Texte.
    """
    if raw_val is None:
        base_vals: List[str] = []
    elif isinstance(raw_val, str):
        base_vals = [raw_val]
    elif isinstance(raw_val, (list, tuple)):
        base_vals = [v for v in raw_val if isinstance(v, str)]
    else:
        base_vals = []

    sectors: List[TargetSector] = []

    for v in base_vals:
        v_s = v.strip()
        if not v_s:
            continue

        # Direkt über Enum-Werte versuchen
        ts = _match_enum_by_value(TargetSector, v_s)
        if ts is not None:
            sectors.append(ts)
            continue

        # Heuristik
        lower = v_s.lower()
        if "house" in lower or "privat" in lower or "haushalt" in lower:
            ts = getattr(TargetSector, "HOUSEHOLDS", None)
        elif "kommunal" in lower or "municipal" in lower or "verwaltung" in lower:
            ts = getattr(TargetSector, "MUNICIPAL_BUILDINGS", None)
        elif "verkehr" in lower or "transport" in lower or "mobil" in lower:
            ts = getattr(TargetSector, "TRANSPORT", None)
        elif "industrie" in lower or "gewerbe" in lower or "wirtschaft" in lower:
            ts = getattr(TargetSector, "INDUSTRY", None)
        elif "cross" in lower or "sektorübergreifend" in lower or "sektoruebergreifend" in lower:
            ts = getattr(TargetSector, "CROSS_SECTORAL", None)
        else:
            ts = getattr(TargetSector, "OTHER", None)

        if ts is not None:
            sectors.append(ts)

    if not sectors:
        other = getattr(TargetSector, "OTHER", None)
        return [other] if other is not None else []

    # Duplikate entfernen, Reihenfolge beibehalten
    seen: set[TargetSector] = set()
    result: List[TargetSector] = []
    for s in sectors:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def _clamp_confidence(raw: Any) -> float:
    try:
        val = float(raw)
    except Exception:  # noqa: BLE001
        return 0.0
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def _infer_funding_bool(raw: Any, instrument_type: Optional[InstrumentType]) -> bool:
    """
    Liefert ein einfaches Bool-Feld "funding_available" für das CSV / NDJSON.
    Die eigentliche Enum-Kodierung kann später ergänzt werden.
    """
    if isinstance(raw, bool):
        return raw

    text = (str(raw) if raw is not None else "").strip().lower()
    if text in {"yes", "ja", "true", "1", "y"}:
        return True
    if text in {"no", "nein", "false", "0", "n"}:
        return False

    # Heuristik: Wenn es sich offensichtlich um ein Förderprogramm handelt → True
    if instrument_type is not None and instrument_type.name == "SUBSIDY_FUNDING_PROGRAM":
        return True

    return False


def _normalize_llm_record(
    raw: Dict[str, Any],
    *,
    fallback_title: str,
    fallback_text: str,
) -> Dict[str, Any]:
    """
    Übersetzt einen einzelnen LLMClient-Record in unser Maßnahmenschema
    (auf Basis von MeasureLabels + Meta-Feldern).

    Erwarteter Input kommt aus LLMClient.extract_batch (inkl. Stub).
    """

    # Titel
    measure_title = (
        raw.get("measure_title")
        or raw.get("title")
        or fallback_title
        or ""
    )

    # Enum-Labels
    policy_area = _parse_policy_area(raw.get("policy_area"))
    instrument_type = _parse_instrument_type(raw.get("instrument_type"))
    target_sectors = _normalize_sectors_enum(
        raw.get("target_sector") or raw.get("target_sectors")
    )
    climate_dimension = _parse_climate_dimension(raw.get("climate_dimension"))
    governance_type = _parse_governance_type(raw.get("governance_type"))

    # Funding (Bool für Downstream, Enum-Field lassen wir vorerst leer)
    funding_bool = _infer_funding_bool(raw.get("funding_available"), instrument_type)

    labels = MeasureLabels(
        policy_area=policy_area,
        instrument_type=instrument_type,
        target_sectors=target_sectors,
        climate_dimension=climate_dimension,
        governance_type=governance_type,
        # FundingAvailable-Enum könnten wir später ergänzen;
        # aktuell bleibt dieses Feld None und wir geben nur das Bool mit aus.
        funding_available=None,
    )

    confidence_score = _clamp_confidence(raw.get("confidence_score", 0.0))
    keyword_hit = bool(raw.get("keyword_hit", False))

    # In ein Dict mit String-Labels für NDJSON/CSV serialisieren
    label_dict = labels.to_dict_str()
    # Sicherstellen, dass unser Bool nicht überschrieben wird
    label_dict.pop("funding_available", None)

    return {
        "measure_title": measure_title,
        **label_dict,
        "funding_available": funding_bool,
        "confidence_score": confidence_score,
        "keyword_hit": keyword_hit,
    }


# -------------------------------------------------
# Öffentliche API, die von src.cli verwendet wird
# -------------------------------------------------


def extract_measures_from_text(
    *,
    client: LLMClient,
    texts: Sequence[str],
    titles: Sequence[str],
    urls: Sequence[str | None],
    scope: str,
) -> List[Dict[str, Any]]:
    """
    Hauptfunktion, die von cmd_auto_label_policies aufgerufen wird.

    Pipeline:
      1) Aufruf von client.extract_batch(...) → LLM oder Stub.
      2) Pro Eintrag Normalisierung in unser Maßnahmenschema:
         - measure_title
         - policy_area
         - instrument_type
         - target_sectors (Liste, als Strings serialisiert)
         - climate_dimension
         - governance_type
         - funding_available (Bool)
         - confidence_score
         - keyword_hit (optional)
    """
    texts_list = list(texts)
    titles_list = list(titles)
    urls_list = list(urls)

    if not (len(texts_list) == len(titles_list) == len(urls_list)):
        raise ValueError("texts, titles und urls müssen die gleiche Länge haben.")

    logger.info(
        "extract_measures_from_text: starte Batch mit %d Kandidaten (scope=%s)",
        len(texts_list),
        scope,
    )

    # Hier passiert ggf. der echte LLM-Call ODER wir landen im Stub-Modus
    raw_measures = client.extract_batch(
        texts=texts_list,
        titles=titles_list,
        urls=urls_list,
        scope=scope,
    )

    if len(raw_measures) != len(texts_list):
        logger.warning(
            "LLMClient.extract_batch hat %d Ergebnisse geliefert, erwartet waren %d. "
            "Trunkiere auf Minimum.",
            len(raw_measures),
            len(texts_list),
        )

    out: List[Dict[str, Any]] = []
    for txt, title, raw in zip(texts_list, titles_list, raw_measures):
        if raw is None:
            raw = {}
        norm = _normalize_llm_record(
            raw,
            fallback_title=title,
            fallback_text=txt,
        )

        # Municipality / document_id / source_url werden in cmd_auto_label_policies ergänzt.
        out.append(norm)

    logger.info("extract_measures_from_text: %d Measures erzeugt.", len(out))
    return out
