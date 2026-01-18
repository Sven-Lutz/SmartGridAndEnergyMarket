from __future__ import annotations

from pathlib import Path
import logging

from src.utils.config import project_root
from .municipal_loader import load_municipalities
from .crawler import crawl_scope

logger = logging.getLogger(__name__)


def run_crawl(
    scope: str,
    data_root: Path | None = None,
    dry_run: bool = False,
    force: bool = False,
    scope_config: dict | None = None,
) -> Path:
    """
    Startet den Crawl für einen Scope.
    """

    # Fallback, falls data_root nicht explizit übergeben wurde
    if data_root is None:
        data_root = project_root() / "data" / scope

    # Konfiguration aus der Scope-Config lesen
    crawl_cfg = (scope_config or {}).get("crawl", {})

    # WICHTIG: Verzeichnis 'meta' übergeben, nicht die Datei!
    municipalities_dir = data_root / "meta"
    municipalities = load_municipalities(municipalities_dir)

    max_municipalities = crawl_cfg.get("max_municipalities", len(municipalities))
    max_pages_per_municipality = crawl_cfg.get("max_pages_per_municipality", 50)
    max_depth = crawl_cfg.get("max_depth", 2)

    output_path = data_root / "staging" / "documents_raw.ndjson"

    logger.info(
        "run_crawl: scope=%s, data_root=%s, max_municipalities=%d, "
        "max_pages_per_municipality=%d, max_depth=%d, dry_run=%s, force=%s",
        scope,
        data_root,
        max_municipalities,
        max_pages_per_municipality,
        max_depth,
        dry_run,
        force,
    )

    if dry_run:
        logger.info("run_crawl im Dry-Run-Modus – es wird nichts gecrawlt.")
        return output_path

    crawl_scope(
        scope=scope,
        municipalities=municipalities,
        output_path=output_path,
        max_municipalities=max_municipalities,
        max_pages_per_municipality=max_pages_per_municipality,
        max_depth=max_depth,
        data_root=data_root,
    )

    return output_path
