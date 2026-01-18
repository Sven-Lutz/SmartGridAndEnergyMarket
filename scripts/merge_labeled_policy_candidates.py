#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import pandas as pd

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
            "Merge: manuell gelabelte Policy-Kandidaten (CSV) mit "
            "documents_extracted.ndjson zu policy_labeled.ndjson."
        )
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
        "--labels-csv",
        default=None,
        help=(
            "Optional: Pfad zur Labels-CSV. "
            "Default: <data_root>/<scope>/labeling/policy_candidates_labeled.csv"
        ),
    )
    parser.add_argument(
        "--extracted-ndjson",
        default=None,
        help=(
            "Optional: Pfad zu documents_extracted.ndjson. "
            "Default: <data_root>/<scope>/extracted/documents_extracted.ndjson"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional: Pfad zur Output-NDJSON-Datei. "
            "Default: <data_root>/<scope>/labeling/policy_labeled.ndjson"
        ),
    )
    parser.add_argument(
        "--id-cols",
        default="municipality,document_id",
        help="Komma-separierte Liste der ID-Spalten, die sowohl in CSV als auch NDJSON vorhanden sind.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose Logging (DEBUG-Level).",
    )
    return parser.parse_args()


def build_label_index(
    df: pd.DataFrame,
    id_cols: Tuple[str, ...],
) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    """
    Baut ein Mapping (id_tuple -> label_dict), wobei label_dict alle Spalten
    außer den ID-Spalten enthält.
    """
    for col in id_cols:
        if col not in df.columns:
            raise ValueError(
                f"ID-Spalte '{col}' nicht in Labels-CSV vorhanden. "
                f"Vorhandene Spalten: {list(df.columns)}"
            )

    label_cols = [c for c in df.columns if c not in id_cols]

    index: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for _, row in df.iterrows():
        key = tuple(row[col] for col in id_cols)
        labels = {c: row[c] for c in label_cols}
        index[key] = labels

    return index


def merge_labeled_candidates(
    labels_csv: Path,
    extracted_ndjson: Path,
    output_path: Path,
    id_cols: Tuple[str, ...],
) -> None:
    logger.info(
        "Starte Merge mit labels_csv='%s', extracted_ndjson='%s', output='%s', id_cols=%s",
        labels_csv,
        extracted_ndjson,
        output_path,
        id_cols,
    )

    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels-CSV nicht gefunden: {labels_csv}")
    if not extracted_ndjson.exists():
        raise FileNotFoundError(f"NDJSON mit extrahierten Dokumenten nicht gefunden: {extracted_ndjson}")

    df = pd.read_csv(labels_csv)
    logger.info("Labels-CSV gelesen: %d Zeilen, Spalten=%s", len(df), list(df.columns))

    label_index = build_label_index(df, id_cols)
    logger.info("Label-Index aufgebaut: %d eindeutige ID(s).", len(label_index))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    docs_total = 0
    docs_labeled = 0
    label_hits = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for doc in iter_ndjson(extracted_ndjson):
            docs_total += 1
            key = tuple(doc.get(col) for col in id_cols)
            labels = label_index.get(key)

            if labels is not None:
                # labels können NaN enthalten -> in None/leer konvertieren
                clean_labels = {}
                for k, v in labels.items():
                    if pd.isna(v):
                        clean_labels[k] = None
                    else:
                        clean_labels[k] = v

                doc_out: Dict[str, Any] = dict(doc)
                doc_out["labels"] = clean_labels
                docs_labeled += 1
                label_hits += 1
            else:
                doc_out = doc

            out_f.write(json.dumps(doc_out, ensure_ascii=False))
            out_f.write("\n")

    logger.info(
        "Merge abgeschlossen. %d Dokumente aus %s gelesen, %d Dokumente mit Labels versehen.",
        docs_total,
        extracted_ndjson,
        docs_labeled,
    )
    logger.info("Output nach %s geschrieben.", output_path)
    logger.info(
        "Label-Index-Hits: %d (Einträge im Index: %d).",
        label_hits,
        len(label_index),
    )


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

    labels_csv = (
        Path(args.labels_csv).resolve()
        if args.labels_csv is not None
        else scope_root / "labeling" / "policy_candidates_labeled.csv"
    )
    extracted_ndjson = (
        Path(args.extracted_ndjson).resolve()
        if args.extracted_ndjson is not None
        else scope_root / "extracted" / "documents_extracted.ndjson"
    )
    output_path = (
        Path(args.output).resolve()
        if args.output is not None
        else scope_root / "labeling" / "policy_labeled.ndjson"
    )

    id_cols = tuple(c.strip() for c in args.id_cols.split(",") if c.strip())
    if not id_cols:
        raise ValueError("id_cols darf nicht leer sein.")

    logger.info(
        "Starte Merge mit scope='%s', labels_csv='%s', extracted_ndjson='%s', "
        "output='%s', id_cols=%s",
        args.scope,
        labels_csv,
        extracted_ndjson,
        output_path,
        id_cols,
    )

    merge_labeled_candidates(
        labels_csv=labels_csv,
        extracted_ndjson=extracted_ndjson,
        output_path=output_path,
        id_cols=id_cols,
    )


if __name__ == "__main__":
    main()
