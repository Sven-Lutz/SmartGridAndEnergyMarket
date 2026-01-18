#!/usr/bin/env python
"""
Debug-Script: Sucht in den extrahierten Dokumenten nach Klima-bezogenen Keywords.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# --- Projektroot auf sys.path legen, damit "import src. ..." funktioniert ---
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent  # .../k3-pipeline

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import project_root  # type: ignore

SCOPE = "de_sample"

KEYWORDS = [
    "klima",
    "klimaschutz",
    "klimaneutral",
    "co2",
    "treibhausgas",
    "energie",
    "wärme",
    "heizung",
    "solar",
    "photovoltaik",
    "emission",
    "förderprogramm",
    "förderung",
]


def main() -> None:
    root = project_root()
    extracted_path = root / "data" / SCOPE / "extracted" / "documents_extracted.ndjson"

    print("Lese:", extracted_path)

    if not extracted_path.exists():
        print("FEHLT:", extracted_path)
        return

    matches = 0

    with extracted_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            doc = json.loads(line)

            title = doc.get("title") or ""
            text = doc.get("text") or ""
            haystack = (title + "\n" + text).lower()

            hit_words = [kw for kw in KEYWORDS if kw in haystack]
            if not hit_words:
                continue

            matches += 1
            print(f"\n=== MATCH {matches:03d} (doc #{i}) =====================")
            print("municipality:", doc.get("municipality_id"))
            print("document_id:", doc.get("document_id"))
            print("url       :", doc.get("source_url") or doc.get("url"))
            print("title     :", title)
            print("keywords  :", ", ".join(hit_words))

            text_flat = text.replace("\n", " ")
            print("text[:300]:", text_flat[:300])

    if matches == 0:
        print("\nKeine Dokumente mit den definierten Keywords gefunden.")
    else:
        print(f"\nFertig. Gefundene Dokumente mit Keywords: {matches}")


if __name__ == "__main__":
    main()
