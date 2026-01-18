# src/crawl/municipal_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import csv

from ..utils.logging_config import get_logger
from ..utils.ndjson_io import read_ndjson_to_list

logger = get_logger(__name__)


def _load_municipalities_csv(path: Path) -> List[Dict[str, Any]]:
    """
    Lädt Kommunen aus municipalities.csv.

    Erwartetes CSV-Schema:
      municipality_id,name,base_url,admin_level,population
    Optional zusätzlich:
      state,confidence_domain,llm_rationale_domain,source
    """
    municipalities: List[Dict[str, Any]] = []

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            municipality_id = row.get("municipality_id")
            base_url = row.get("base_url")
            if not municipality_id or not base_url:
                logger.warning("Überspringe Zeile ohne municipality_id/base_url: %r", row)
                continue

            name = row.get("name") or row.get("official_name") or municipality_id

            # Kompatible Struktur zu alter NDJSON-Variante
            m: Dict[str, Any] = {
                "municipality_id": municipality_id,
                "official_name": row.get("official_name") or name,
                "name": name,
                "state": row.get("state"),
                "base_url": base_url,
                "admin_level": row.get("admin_level"),
                "population": int(row["population"]) if row.get("population") else None,
                "confidence_domain": row.get("confidence_domain"),
                "llm_rationale_domain": row.get("llm_rationale_domain"),
                "source": row.get("source") or "municipalities.csv",
            }
            municipalities.append(m)

    logger.info(
        "load_municipalities (csv): %d Kommunen aus %s geladen.",
        len(municipalities),
        path,
    )
    return municipalities


def _load_municipalities_ndjson(path: Path) -> List[Dict[str, Any]]:
    municipalities = read_ndjson_to_list(path)
    logger.info(
        "load_municipalities (ndjson): %d Kommunen aus %s geladen.",
        len(municipalities),
        path,
    )
    return municipalities


def load_municipalities(meta_root: Path) -> List[Dict[str, Any]]:
    """
    Lädt die Kommunenliste aus meta/municipalities.csv (bevorzugt)
    oder – falls nicht vorhanden – aus meta/municipalities.ndjson.

    CSV und NDJSON werden auf ein gemeinsames Dict-Schema abgebildet.
    """
    csv_path = meta_root / "municipalities.csv"
    ndjson_path = meta_root / "municipalities.ndjson"

    if csv_path.exists():
        return _load_municipalities_csv(csv_path)

    if ndjson_path.exists():
        logger.warning(
            "municipalities.csv nicht gefunden, fallback auf NDJSON: %s",
            ndjson_path,
        )
        return _load_municipalities_ndjson(ndjson_path)

    logger.warning(
        "Keine Kommunendatei gefunden (weder %s noch %s).",
        csv_path,
        ndjson_path,
    )
    return []
