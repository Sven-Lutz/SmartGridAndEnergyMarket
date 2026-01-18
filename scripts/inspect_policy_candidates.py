#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Projekt-Root auf sys.path legen
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.ndjson import iter_ndjson  # zentrale NDJSON-Hilfe

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect: policy_candidates.ndjson."
    )
    parser.add_argument(
        "--scope",
        required=True,
        help="Daten-Scope (z.B. 'de_by'). Erwartet Daten unter data/<scope>/extracted/.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Optionaler Daten-Root. Default: PROJECT_ROOT / 'data'.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional: expliziter Pfad zur policy_candidates.ndjson. "
             "Default: <data_root>/<scope>/extracted/policy_candidates.ndjson",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Max. Anzahl der anzuzeigenden Dokumente.",
    )
    parser.add_argument(
        "--municipality",
        default=None,
        help="Optional: nur Dokumente dieser municipality anzeigen.",
    )
    parser.add_argument(
        "--contains",
        default=None,
        help="Optional: nur Dokumente, deren Text oder Titel diesen String enthält.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose Logging (DEBUG-Level).",
    )
    return parser.parse_args()


def format_doc(doc: Dict[str, Any], idx: int, text_preview_len: int = 200) -> str:
    municipality = doc.get("municipality") or ""
    document_id = doc.get("document_id") or ""
    url = doc.get("url") or doc.get("source_url") or ""
    title = doc.get("title") or ""
    text = (doc.get("text") or "").replace("\n", " ")
    preview = text[:text_preview_len]

    lines = [
        f"--- POLICY DOC {idx:03d} -----------------------------",
        f"municipality: {municipality}",
        f"document_id : {document_id}",
        f"url         : {url}",
        f"title       : {title}",
        f"text[:{text_preview_len}]  : {preview}",
        "",
    ]
    return "\n".join(lines)


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

    logger.info("Lese Policy-Kandidaten aus: %s", input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Eingabedatei nicht gefunden: {input_path}")

    shown = 0
    for doc in iter_ndjson(input_path):
        if args.municipality and (doc.get("municipality") != args.municipality):
            continue

        if args.contains:
            haystack = (doc.get("title") or "") + " " + (doc.get("text") or "")
            if args.contains not in haystack:
                continue

        print(format_doc(doc, shown))
        shown += 1
        if shown >= args.limit:
            break

    logger.info("--- Ende, %d Dokument(e) angezeigt. ---", shown)


if __name__ == "__main__":
    main()
