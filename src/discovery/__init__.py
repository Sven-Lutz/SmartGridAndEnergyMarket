from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..utils.logging_config import get_logger
from .municipality_discovery import discover_municipalities

logger = get_logger(__name__)


def run_discovery(
    scope: str,
    scope_config: Dict[str, Any],
    data_root: Path,
    dry_run: bool = False,
) -> None:
    """
    High-Level-Einstiegspunkt für die Municipality Discovery.

    Wird von src/cli.py aufgerufen.
    Verantwortlich für:
      - Pfade für Output bestimmen
      - ggf. weitere Konfigurationen laden
      - discover_municipalities(...) delegieren
    """
    logger.info(
        "run_discovery gestartet für scope='%s', dry_run=%s",
        scope,
        dry_run,
    )

    out_path = data_root / "municipalities.ndjson"

    discover_municipalities(
        scope=scope,
        scope_config=scope_config,
        output_path=out_path,
        dry_run=dry_run,
    )

    logger.info(
        "run_discovery beendet für scope='%s'. Output: %s",
        scope,
        out_path,
    )
