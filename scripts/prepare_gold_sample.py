#!/usr/bin/env python
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

# --- Projektwurzel auf sys.path setzen ---
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import project_root  # noqa: E402


def main() -> None:
    scope = "de_sample"
    mode = "baseline"
    n_sample = 50  # Anzahl Zeilen für Gold-Annotation

    data_root = project_root() / "data" / scope / "classified"
    input_file = data_root / f"measures_{mode}.csv"
    output_file = data_root / f"measures_{mode}_gold_sample.csv"

    print(f"Lese: {input_file}")
    if not input_file.exists():
        print(f"FEHLER: {input_file} existiert nicht.")
        return

    with input_file.open("r", encoding="utf-8", newline="") as f_in:
        reader = list(csv.DictReader(f_in))

    if not reader:
        print("Keine Zeilen in der Input-CSV.")
        return

    # Zufällige Stichprobe (oder nimm einfach die ersten n_sample Zeilen, wenn du deterministisch willst)
    sample = reader[:n_sample]  # deterministisch
    # sample = random.sample(reader, min(n_sample, len(reader)))  # zufällig

    # vorhandene Spalten
    base_fields = list(sample[0].keys())

    # neue Gold-Standard-Spalten
    gold_fields = [
        "is_measure_gold",      # 1 oder 0
        "measure_type_gold",    # GRANT / REGULATION / ADVICE / OTHER / NONE
        "sector_gold",          # E_MOBILITY / RENEWABLES / ... / NONE
        "comment_gold",         # Freitext
    ]

    fieldnames = base_fields + gold_fields

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in sample:
            for gf in gold_fields:
                row[gf] = ""  # leere Spalten für manuelle Annotation
            writer.writerow(row)

    print(f"Fertig. Gold-Sample nach {output_file} geschrieben.")
    print("Bitte diese Datei in Excel/LibreOffice öffnen und die *_gold-Spalten manuell annotieren.")


if __name__ == "__main__":
    main()
