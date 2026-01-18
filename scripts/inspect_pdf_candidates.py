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
        description=(
            "Inspect: PDF-Kandidaten in documents_extracted.ndjson "
            "oder policy_candidates.ndjson."
        )
    )
    parser.add_argument(
        "--scope",
        required=True,
        help="Daten-Scope (z.B. 'de_by').",
    )
    parser.add_argument(
        "--source",
        choices=["extracted", "policy"],
        default="extracted",
        help=(
            "'extracted' = documents_extracted.ndjson, "
            "'policy' = policy_candidates.ndjson"
        ),
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Optionaler Daten-Root. Default: PROJECT_ROOT / 'data'.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional: expliziter Pfad zur NDJSON-Datei.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max. Anzahl der anzuzeigenden PDF-Dokumente.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose Logging (DEBUG-Level).",
    )
    return parser.parse_args()


def is_pdf_doc(doc: Dict[str, Any]) -> bool:
    mime_type = (doc.get("mime_type") or "").lower()
    url = (doc.get("url") or doc.get("source_url") or "").lower()
    if "pdf" in mime_type:
        return True
    if url.split("?", 1)[0].endswith(".pdf"):
        return True
    return False


def format_pdf_doc(doc: Dict[str, Any], idx: int, text_preview_len: int = 500) -> str:
    municipality = doc.get("municipality") or ""
    document_id = doc.get("document_id") or ""
    url = doc.get("url") or doc.get("source_url") or ""
    title = doc.get("title") or ""
    mime_type = doc.get("mime_type") or ""
    text = (doc.get("text") or "").replace("\n", " ")
    preview = text[:text_preview_len]

    lines = [
        f"--- PDF DOC {idx:03d} -----------------------------",
        f"municipality: {municipality}",
        f"document_id : {document_id}",
        f"url         : {url}",
        f"title       : {title}",
        f"mime_type   : {mime_type}",
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

    if args.input is not None:
        input_path = Path(args.input).resolve()
    else:
        if args.source == "extracted":
            input_path = scope_root / "extracted" / "documents_extracted.ndjson"
        else:
            input_path = scope_root / "extracted" / "policy_candidates.ndjson"

    logger.info(
        "Starte inspect_pdfs mit scope='%s', source='%s', input='%s', limit=%d",
        args.scope,
        args.source,
        input_path,
        args.limit,
    )

    logger.info("Lese NDJSON aus: %s", input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Eingabedatei nicht gefunden: {input_path}")

    shown = 0
    total_pdf = 0

    for doc in iter_ndjson(input_path):
        if not is_pdf_doc(doc):
            continue

        if shown < args.limit:
            print(format_pdf_doc(doc, shown))
        shown += 1
        total_pdf += 1
        if shown >= args.limit and total_pdf > args.limit:
            # weiterzählen, aber nichts mehr ausgeben
            continue

    logger.info(
        "--- Ende, %d PDF-Dokument(e) angezeigt (von insgesamt %d Einträgen). ---",
        min(shown, args.limit),
        total_pdf,
    )


if __name__ == "__main__":
    main()
