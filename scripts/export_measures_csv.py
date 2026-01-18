#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# --- Fix: Projektwurzel zum sys.path hinzufügen ---
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]  # ein Ordner über "scripts/"
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import project_root  # funktioniert jetzt


def main() -> None:
    scope = "de_sample"
    mode = "baseline"

    data_root = project_root() / "data" / scope
    input_file = data_root / "classified" / f"measures_{mode}.ndjson"
    output_file = data_root / "classified" / f"measures_{mode}.csv"

    print(f"Lese Measures aus: {input_file}")
    if not input_file.exists():
        print(f"FEHLER: {input_file} existiert nicht.")
        return

    rows = []
    with input_file.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(obj)

    if not rows:
        print("Keine Maßnahmen gefunden – CSV bleibt leer.")
        return

    fieldnames = [
        "municipality_id",
        "document_id",
        "source_url",
        "measure_title",
        "measure_type",
        "sector",
        "target_group",
        "amount_min",
        "amount_max",
        "application_mode",
        "digitalization_level",
        "extraction_mode",
        "baseline_keywords",
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for r in rows:
            bk = r.get("baseline_keywords")
            if isinstance(bk, list):
                r["baseline_keywords"] = ", ".join(bk)
            writer.writerow({k: r.get(k) for k in fieldnames})

    print(f"Fertig. {len(rows)} Maßnahmen nach {output_file} geschrieben.")


if __name__ == "__main__":
    main()
