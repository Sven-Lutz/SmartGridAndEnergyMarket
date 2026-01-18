#!/usr/bin/env python
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from ..utils.config import project_root

logger = logging.getLogger(__name__)


def _read_ndjson(path: Path) -> List[Dict[str, Any]]:
    """
    Liest eine NDJSON-Datei in eine Liste von Dicts ein.
    """
    items: List[Dict[str, Any]] = []

    if not path.exists():
        logger.warning("Evaluations-Input %s existiert nicht.", path)
        return items

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception as e:
                logger.warning(
                    "Konnte Zeile in %s nicht parsen (%s). Zeile wird übersprungen.",
                    path,
                    e,
                )
    return items


def _get_measures_path(data_root: Path, mode: str) -> Path:
    """
    Gibt den Pfad zur Measures-Datei für den gegebenen Modus zurück.
    """
    classified_dir = data_root / "classified"

    if mode == "baseline":
        return classified_dir / "measures_baseline.ndjson"
    elif mode == "llm":
        return classified_dir / "measures_llm.ndjson"
    elif mode == "both":
        # Optional: hier könnte man zwei Dateien auswerten.
        # Fürs erste verwenden wir die Baseline-Datei als Default.
        return classified_dir / "measures_baseline.ndjson"
    else:
        # Sollte durch validate_mode in cli.py bereits abgefangen sein
        raise ValueError(f"Unbekannter Evaluationsmodus: {mode!r}")


def run_eval(
    scope: str,
    mode: str,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    """
    Führt eine einfache Evaluation aus.

    Aktuell:
    - Liest die klassifizierten Maßnahmen aus data/<scope>/classified/
    - Zählt die Anzahl der Maßnahmen
    - Schreibt eine kleine JSON-Statistik nach data/<scope>/eval/eval_<mode>.json
    """

    data_root = project_root() / "data" / scope
    eval_dir = data_root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    measures_path = _get_measures_path(data_root, mode)
    logger.info(
        "Starte Evaluation für scope='%s', mode='%s'. "
        "Lese Measures aus %s",
        scope,
        mode,
        measures_path,
    )

    if dry_run:
        logger.info(
            "[dry-run] Würde Evaluation ausführen und Statistiken nach %s schreiben.",
            eval_dir,
        )
        return

    measures = _read_ndjson(measures_path)
    num_measures = len(measures)

    stats: Dict[str, Any] = {
        "scope": scope,
        "mode": mode,
        "num_measures": num_measures,
        "input_file": str(measures_path),
    }

    # Optional: später kannst du hier Precision/Recall etc. ergänzen,
    # sobald es Gold-Labels gibt.
    eval_out_path = eval_dir / f"eval_{mode}.json"
    with eval_out_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(
        "Evaluation abgeschlossen für scope='%s', mode='%s'. "
        "Anzahl Maßnahmen: %d. Statistik geschrieben nach %s",
        scope,
        mode,
        num_measures,
        eval_out_path,
    )
