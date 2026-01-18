#!/usr/bin/env python
"""
scripts/discover_seeds.py

Verwendung:
    python scripts/discover_seeds.py --scope de_by --max-seeds 5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict
import sys

# ---------------------------------------------------------------------------
# Projekt-Root auf sys.path setzen, damit 'src' importierbar ist
# ---------------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.crawl.seed_discovery import discover_seeds_for_municipality
from src.crawl.municipal_loader import load_municipalities
from src.utils.config import project_root  # gleiche Util wie in anderen Scripts

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover climate-related seeds per municipality.")
    parser.add_argument(
        "--scope",
        required=True,
        help="Name des Scopes (z. B. de_by, de_sample, ...)",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=5,
        help="Maximale Anzahl Seeds pro Kommune (Default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur loggen, aber nichts schreiben.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    data_root = project_root() / "data" / args.scope
    meta_dir = data_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # municipalities.csv (bevorzugt) oder municipalities.ndjson
    logger.info("Lade Municipalities aus: %s", meta_dir)
    municipalities: list[Dict[str, Any]] = load_municipalities(meta_dir)
    if not municipalities:
        raise SystemExit(f"Keine Municipalities gefunden in {meta_dir}")

    out_path = meta_dir / "discovered_seeds.ndjson"
    if args.dry_run:
        logger.info("Dry-Run aktiviert: es wird NICHT nach %s geschrieben.", out_path)
    else:
        logger.info("Seeds werden nach %s geschrieben.", out_path)

    lines_out: list[Dict[str, Any]] = []

    for m in municipalities:
        muni_id = m.get("municipality_id") or m.get("id")
        base_url = m.get("base_url") or m.get("homepage_url")

        if not muni_id or not base_url:
            logger.warning(
                "Überspringe Eintrag ohne municipality_id oder base_url: %s",
                m,
            )
            continue

        candidates = discover_seeds_for_municipality(
            municipality_id=muni_id,
            base_url=base_url,
            max_seeds=args.max_seeds,
        )

        record: Dict[str, Any] = {
            "municipality_id": muni_id,
            "base_url": base_url,
            "discovered_seeds": [c.url for c in candidates],
            "debug": {
                "anchor_texts": [c.anchor_text for c in candidates],
                "scores": [c.score for c in candidates],
            },
        }
        lines_out.append(record)

        logger.info(
            "Seeds für %s: %s",
            muni_id,
            ", ".join(record["discovered_seeds"]) if record["discovered_seeds"] else "(keine gefunden)",
        )

    if not args.dry_run:
        with out_path.open("w", encoding="utf-8") as f:
            for rec in lines_out:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        logger.info(
            "Fertig. Für %d Kommunen Seeds nach %s geschrieben.",
            len(lines_out),
            out_path,
        )
    else:
        logger.info("Dry-Run beendet. %d Kommunen verarbeitet.", len(lines_out))


if __name__ == "__main__":
    main()
