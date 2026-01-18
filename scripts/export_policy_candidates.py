#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Projekt-Root auf sys.path legen
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.policy_filter import is_policy_relevant, DEFAULT_CONFIG  # type: ignore[import]
from src.utils.ndjson import iter_ndjson  # zentrale NDJSON-Hilfe

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export policy-relevante Dokumente als NDJSON."
    )
    parser.add_argument(
        "--scope",
        required=True,
        help="Daten-Scope (z.B. 'de_sample'). Erwartet Daten unter data/<scope>/extracted/.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Optionaler Daten-Root. Default: PROJECT_ROOT / 'data'.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional: expliziter Pfad zur documents_extracted.ndjson. "
             "Default: <data_root>/<scope>/extracted/documents_extracted.ndjson",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional: expliziter Pfad zur policy_candidates.ndjson. "
             "Default: <data_root>/<scope>/extracted/policy_candidates.ndjson",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur zählen und auf stdout loggen, keine Datei schreiben.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose Logging (DEBUG-Level).",
    )
    return parser.parse_args()


def export_policy_candidates(
    input_path: Path,
    output_path: Path,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Filtert policy-relevante Dokumente aus documents_extracted.ndjson
    und schreibt sie als NDJSON nach policy_candidates.ndjson.

    Returns:
        (total_docs, kept_docs)
    """
    logger.info("Lese extrahierte Dokumente aus: %s", input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Eingabedatei nicht gefunden: {input_path}")

    total = 0
    kept = 0

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = output_path.open("w", encoding="utf-8")
    else:
        out_f = None

    try:
        for doc in iter_ndjson(input_path):
            total += 1

            text = doc.get("text") or ""
            title = doc.get("title") or ""
            url = doc.get("source_url") or doc.get("url") or ""

            if not is_policy_relevant(text=text, title=title, url=url, cfg=DEFAULT_CONFIG):
                continue

            kept += 1
            if not dry_run and out_f is not None:
                out_f.write(json.dumps(doc, ensure_ascii=False))
                out_f.write("\n")
    finally:
        if out_f is not None:
            out_f.close()

    logger.info(
        "Fertig. %d von %d Dokumenten als policy-relevant exportiert (%.1f%%).",
        kept,
        total,
        100.0 * kept / total if total > 0 else 0.0,
    )

    if not dry_run:
        logger.info("Output geschrieben nach: %s", output_path)

    return total, kept


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
        else scope_root / "extracted" / "documents_extracted.ndjson"
    )
    output_path = (
        Path(args.output).resolve()
        if args.output is not None
        else scope_root / "extracted" / "policy_candidates.ndjson"
    )

    logger.info(
        "Starte Export der Policy-Kandidaten mit scope='%s', input='%s', output='%s', dry_run=%s",
        args.scope,
        input_path,
        output_path,
        args.dry_run,
    )

    export_policy_candidates(
        input_path=input_path,
        output_path=output_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
