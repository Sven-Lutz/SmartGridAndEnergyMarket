from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Any, Tuple

from src.analysis.measure_schema import MeasureAnalysisRow
from src.crawl.municipal_loader import load_municipalities  # existiert bereits
import logging

logger = logging.getLogger(__name__)


def _iter_ndjson(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_documents_by_id(documents_path: Path) -> Dict[Tuple[str, str], dict]:
    """
    Lädt documents_extracted.ndjson in ein Dict:

        (municipality_id, document_id) -> record
    """
    logger.info("Lade documents_extracted aus %s", documents_path)
    mapping: Dict[Tuple[str, str], dict] = {}
    for rec in _iter_ndjson(documents_path):
        muni_id = rec.get("municipality_id")
        doc_id = rec.get("document_id")
        if muni_id is None or doc_id is None:
            continue
        mapping[(muni_id, doc_id)] = rec
    logger.info("Geladene Dokumente (extracted): %d", len(mapping))
    return mapping


def _load_municipalities_by_id(meta_dir: Path) -> Dict[str, Dict[str, Any]]:
    logger.info("Lade municipalities aus %s", meta_dir)
    muni_list = load_municipalities(meta_dir)
    logger.info("Geladene Municipalities: %d", len(muni_list))
    by_id: Dict[str, Dict[str, Any]] = {}
    for m in muni_list:
        m_id = m.get("municipality_id")
        if m_id:
            by_id[m_id] = m
    return by_id


def _get_municipality_name(m: dict) -> str:
    """
    Versucht, einen sinnvollen Anzeigenamen zu finden.
    Fallback: municipality_id.
    """
    return (
        m.get("municipality_name")
        or m.get("name")
        or m.get("municipality_id")
        or ""
    )


def _build_measure_rows_for_scope(
    *,
    scope: str,
    measures_path: Path,
    documents_path: Path,
    municipalities_dir: Path,
    country_code: str = "DE",
    state_code: str = "BY",
    extraction_version: str = "doc_extractor_v1",
    classification_version: str = "baseline_rules_v1",
    text_excerpt_len: int = 500,
) -> List[MeasureAnalysisRow]:
    """
    Erzeugt MeasureAnalysisRow-Objekte für einen Scope, indem
    measures_*.ndjson mit documents_extracted und Municipalities-Metadaten
    (aus dem Meta-Verzeichnis) gejoint wird.
    """

    municipalities = _load_municipalities_by_id(municipalities_dir)
    docs_by_id = _load_documents_by_id(documents_path)

    rows: List[MeasureAnalysisRow] = []

    # Zähler für measure_local_id pro (municipality_id, document_id)
    local_id_counter: Dict[Tuple[str, str], int] = defaultdict(int)

    logger.info("Lese Measures aus %s", measures_path)
    for m in _iter_ndjson(measures_path):
        municipality_id = m.get("municipality_id")
        document_id = m.get("document_id")
        source_url = m.get("source_url", "")

        if municipality_id is None or document_id is None:
            logger.warning(
                "Measure ohne municipality_id/document_id gefunden, überspringe: %s",
                m,
            )
            continue

        muni_meta = municipalities.get(municipality_id, {})
        municipality_name = _get_municipality_name(muni_meta)

        doc_rec = docs_by_id.get((municipality_id, document_id), {})

        # measure_local_id inkrementell pro (municipality_id, document_id)
        key = (municipality_id, document_id)
        local_id = local_id_counter[key]
        local_id_counter[key] += 1

        measure_uid = f"{scope}::{municipality_id}::{document_id}::{local_id}"

        # Dokumentfelder
        crawl_timestamp = doc_rec.get("crawl_timestamp")
        depth = doc_rec.get("depth")
        referrer_url = doc_rec.get("referrer_url")
        http_status = doc_rec.get("http_status")
        mime_type = doc_rec.get("mime_type")

        document_title = doc_rec.get("title") or ""
        full_text = doc_rec.get("text") or ""
        text_excerpt = full_text[:text_excerpt_len]

        # Maßnahmenfelder (kommen aus classified/measures_*.ndjson)
        measure_title = m.get("measure_title") or document_title or ""

        policy_area = m.get("policy_area") or "unspecified"
        instrument_type = m.get("instrument_type") or "other"
        target_sectors = m.get("target_sectors") or []
        climate_dimension = m.get("climate_dimension") or "unspecified"
        governance_type = m.get("governance_type") or "local_municipality"
        funding_available = bool(m.get("funding_available", False))

        label_source = m.get("label_source") or classification_version
        confidence_score = float(m.get("confidence_score", 0.0))

        keyword_hit = bool(
            m.get("keyword_hit", doc_rec.get("keyword_hit", False))
        )

        row = MeasureAnalysisRow(
            scope=scope,
            country_code=country_code,
            state_code=state_code,
            municipality_id=municipality_id,
            municipality_name=municipality_name,
            measure_local_id=local_id,
            measure_uid=measure_uid,
            document_id=document_id,
            source_url=source_url,
            crawl_timestamp=crawl_timestamp,
            depth=depth,
            referrer_url=referrer_url,
            http_status=http_status,
            mime_type=mime_type,
            document_title=document_title,
            measure_title=measure_title,
            text_excerpt=text_excerpt,
            policy_area=policy_area,
            instrument_type=instrument_type,
            target_sectors=target_sectors,
            climate_dimension=climate_dimension,
            governance_type=governance_type,
            funding_available=funding_available,
            label_source=label_source,
            confidence_score=confidence_score,
            extraction_version=extraction_version,
            classification_version=classification_version,
            keyword_hit=keyword_hit,
        )

        rows.append(row)

    logger.info("Erzeugte MeasureAnalysisRow-Objekte: %d", len(rows))
    return rows


def write_measure_analysis_csv(rows: List[MeasureAnalysisRow], output_csv: Path) -> None:
    """
    Schreibt die MeasureAnalysisRow-Liste als CSV.
    Listenfelder (target_sectors) werden als '|'-separierte Strings serialisiert.
    """
    if not rows:
        logger.warning("Keine MeasureAnalysisRow-Objekte übergeben – schreibe leeres CSV.")
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            pass
        return

    # Feldnamen direkt aus der Dataclass ableiten
    fieldnames = [f.name for f in MeasureAnalysisRow.__dataclass_fields__.values()]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Schreibe Measure-Analysis-CSV nach %s", output_csv)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            d = row.__dict__.copy()

            # target_sectors: Liste -> '|'-String
            ts = d.get("target_sectors") or []
            if isinstance(ts, list):
                d["target_sectors"] = "|".join(ts)
            else:
                d["target_sectors"] = str(ts)

            writer.writerow(d)


