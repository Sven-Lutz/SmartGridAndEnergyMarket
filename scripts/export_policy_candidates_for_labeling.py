#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

# Projekt-Root auf sys.path legen
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.ndjson import iter_ndjson  # zentrale NDJSON-Hilfe

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exportiert policy_candidates.ndjson als CSV für manuelles Labeling."
    )
    parser.add_argument(
        "--scope",
        required=True,
        help="Daten-Scope (z.B. 'de_by').",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Optionaler Daten-Root. Default: PROJECT_ROOT / 'data'.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Optional: expliziter Pfad zur policy_candidates.ndjson. "
            "Default: <data_root>/<scope>/extracted/policy_candidates.ndjson"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional: Pfad zur Output-CSV. "
            "Default: <data_root>/<scope>/labeling/policy_candidates_for_labeling.csv"
        ),
    )
    parser.add_argument(
        "--max-text-len",
        type=int,
        default=2000,
        help="Maximale Länge des Textfeldes pro Zeile (für CSV gekürzt).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional: Stichprobengröße (0 = alle Dokumente exportieren).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random Seed für die Stichprobenziehung.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose Logging (DEBUG-Level).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    data_root = (
        Path(args.data_root).resolve()
        if args.data_root is not None
        else (PROJECT_ROOT / "data")
    )
    scope_root = data_root / args.scope

    input_path = (
        Path(args.input).resolve()
        if args.input is not None
        else scope_root / "extracted" / "policy_candidates.ndjson"
    )

    if args.output is not None:
        output_path = Path(args.output).resolve()
    else:
        output_path = scope_root / "labeling" / "policy_candidates_for_labeling.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starte CSV-Export für Labeling mit scope='%s', input='%s', output='%s', "
        "max_text_len=%d, sample_size=%d, random_seed=%d",
        args.scope,
        input_path,
        output_path,
        args.max_text_len,
        args.sample_size,
        args.random_seed,
    )

    logger.info("Lese Policy-Kandidaten aus: %s", input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Eingabedatei nicht gefunden: {input_path}")

    docs: List[Dict[str, Any]] = list(iter_ndjson(input_path))
    total = len(docs)
    logger.info("Insgesamt %d Policy-Kandidaten eingelesen.", total)

    if args.sample_size > 0 and args.sample_size < total:
        rnd = random.Random(args.random_seed)
        docs = rnd.sample(docs, args.sample_size)
        logger.info(
            "Ziehe Stichprobe von %d Dokumenten aus %d (Seed=%d).",
            args.sample_size,
            total,
            args.random_seed,
        )

    # CSV-Header inkl. Label-Spalten
    fieldnames = [
        "municipality",
        "document_id",
        "url",
        "title",
        "text",
        "mime_type",
        # Label-Spalten für manuelle Annotation
        "label_is_policy_relevant",   # 0/1 oder leer
        "label_policy_scope",         # z.B. 'municipal', 'regional', 'national'
        "label_instrument_type",      # z.B. 'plan', 'ordinance', ...
        "label_target_sector",        # z.B. 'buildings', 'transport', ...
        "label_notes",                # Freitext
    ]

    rows_written = 0
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for doc in docs:
            municipality = doc.get("municipality") or ""
            document_id = doc.get("document_id") or ""
            url = doc.get("url") or doc.get("source_url") or ""
            title = doc.get("title") or ""
            mime_type = doc.get("mime_type") or ""
            text = (doc.get("text") or "").replace("\r", " ").replace("\n", " ")
            if args.max_text_len > 0 and len(text) > args.max_text_len:
                text = text[: args.max_text_len]

            row = {
                "municipality": municipality,
                "document_id": document_id,
                "url": url,
                "title": title,
                "text": text,
                "mime_type": mime_type,
                "label_is_policy_relevant": "",
                "label_policy_scope": "",
                "label_instrument_type": "",
                "label_target_sector": "",
                "label_notes": "",
            }
            writer.writerow(row)
            rows_written += 1

    logger.info(
        "CSV-Export abgeschlossen. %d Dokumente nach %s geschrieben (von %d gelesen).",
        rows_written,
        output_path,
        total,
    )


if __name__ == "__main__":
    main()
