# src/utils/ndjson.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

logger = logging.getLogger(__name__)


def iter_ndjson(path: Path) -> Iterator[Dict[str, Any]]:
    """
    Robustes Einlesen einer NDJSON-Datei.

    - Überspringt leere oder Whitespace-Zeilen
    - Überspringt ungültige JSON-Zeilen, loggt aber eine Warnung
    - Liefert pro gültiger Zeile ein Dict[str, Any] zurück
    """
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            # `raw_line` ist laut Typ immer `str`, also nicht Optional
            line = raw_line.strip()

            # Leere/Whitespace-Zeilen überspringen
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Überspringe ungültige JSON-Zeile %d in %s: %s",
                    line_no,
                    path,
                    exc,
                )
                continue

            # Nur Dicts durchlassen (optional, aber praktisch)
            if isinstance(obj, dict):
                yield obj
            else:
                logger.warning(
                    "Überspringe JSON-Zeile %d in %s: erwartetes Objekt vom Typ dict, erhalten: %s",
                    line_no,
                    path,
                    type(obj).__name__,
                )
