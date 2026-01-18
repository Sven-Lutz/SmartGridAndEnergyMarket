from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Union
import json


JsonDict = Dict[str, Any]
PathLike = Union[str, Path]


def read_ndjson(path: PathLike) -> Iterator[JsonDict]:
    """
    Liest eine NDJSON-Datei zeilenweise und gibt Dictionaries zurück.

    Nutzung:
        for record in read_ndjson("file.ndjson"):
            ...
    """
    p = Path(path)
    if not p.exists():
        return iter(())  # leere Iteration

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_ndjson(
    path: PathLike,
    records: Iterable[JsonDict],
    append: bool = False,
) -> None:
    """
    Schreibt eine Folge von Dictionaries als NDJSON.
    Standard ist Überschreiben, mit append=True wird angehängt.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"

    with p.open(mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def append_ndjson_record(path: PathLike, record: JsonDict) -> None:
    """
    Hängt genau einen Datensatz an eine NDJSON-Datei an.
    """
    write_ndjson(path, [record], append=True)


def read_ndjson_to_list(path: PathLike) -> List[JsonDict]:
    """
    Bequemer Helper, der eine NDJSON-Datei vollständig in eine Liste lädt.
    Achtung bei sehr großen Dateien.
    """
    return list(read_ndjson(path))
