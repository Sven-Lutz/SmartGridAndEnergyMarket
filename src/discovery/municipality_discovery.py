from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..utils.logging_config import get_logger
from ..utils.ndjson_io import write_ndjson

logger = get_logger(__name__)


def discover_municipalities(
    scope: str,
    scope_config: Dict[str, Any],
    output_path: Path,
    dry_run: bool = False,
) -> None:
    """
    Kernlogik für die Municipality Discovery.

    Aktuell nur Skeleton:
      - Für scope 'de_sample' könnte hier testweise eine feste Liste geschrieben werden.
      - Später:
          * Quelllisten (Open Data, Wikipedia, …) abrufen
          * LLM zur Normalisierung und Domain-Resolution nutzen
    """
    logger.info(
        "discover_municipalities gestartet für scope='%s', dry_run=%s",
        scope,
        dry_run,
    )

    if dry_run:
        logger.info(
            "[DRY RUN] Würde Municipality Discovery für scope='%s' durchführen und nach %s schreiben.",
            scope,
            output_path,
        )
        return

    # Minimaler Platzhalter – später durch echte Logik ersetzt.
    records: List[Dict[str, Any]] = []

    if scope == "de_sample":
        # Beispielhafte statische Einträge für Entwicklung
        records = [
            {
                "municipality_id": "karlsruhe",
                "official_name": "Stadt Karlsruhe",
                "state": "Baden-Württemberg",
                "base_url": "https://www.karlsruhe.de",
                "confidence_domain": 0.99,
                "llm_rationale_domain": "Platzhalter: manuell gesetzt für de_sample.",
                "source": "manual_seed_de_sample",
            },
            {
                "municipality_id": "muenchen",
                "official_name": "Landeshauptstadt München",
                "state": "Bayern",
                "base_url": "https://www.muenchen.de",
                "confidence_domain": 0.99,
                "llm_rationale_domain": "Platzhalter: manuell gesetzt für de_sample.",
                "source": "manual_seed_de_sample",
            },
        ]
    else:
        logger.warning(
            "discover_municipalities: Für scope='%s' ist aktuell noch keine Discovery-Logik implementiert.",
            scope,
        )

    write_ndjson(output_path, records, append=False)

    logger.info(
        "discover_municipalities beendet für scope='%s'. %d Einträge geschrieben.",
        scope,
        len(records),
    )
