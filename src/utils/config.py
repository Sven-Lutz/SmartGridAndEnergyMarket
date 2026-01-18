# src/utils/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from .logging_config import get_logger

logger = get_logger(__name__)


# -------------------------------------------------
# Projektwurzel bestimmen
# -------------------------------------------------

def project_root() -> Path:
    """
    Liefert das Projektwurzelverzeichnis (Ordner, der src/ enthält).
    Beispiel: .../k3-pipeline
    """
    root = Path(__file__).resolve().parents[2]
    return root


# -------------------------------------------------
# YAML laden
# -------------------------------------------------

def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        logger.error("YAML-Datei nicht gefunden: %s", path)
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rekursives Mergen von Dictionaries.
    Werte in 'override' überschreiben 'base'.
    """
    result = dict(base)
    for k, v in override.items():
        if (
            k in result
            and isinstance(result[k], dict)
            and isinstance(v, dict)
        ):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# -------------------------------------------------
# Scope-Konfiguration laden (configs/scopes.yml)
# -------------------------------------------------

def load_scope_config(scope: str) -> Dict[str, Any]:
    """
    Lädt die Konfiguration eines Scopes aus configs/scopes.yml.

    Erwartete Struktur in scopes.yml:

    scopes:
      de_by:
        dataset_id: "de_by"
        base_dir: "data/de_by"
        sample_dataset_id: "de_sample"
        municipalities_config: "de_by.yml"
        ...
    """
    cfg_path = project_root() / "configs" / "scopes.yml"
    cfg = _load_yaml(cfg_path)

    scopes = cfg.get("scopes", {})
    if scope not in scopes:
        available = ", ".join(sorted(scopes.keys()))
        raise ValueError(
            f"Scope '{scope}' ist in scopes.yml nicht definiert. "
            f"Verfügbare Scopes: {available}"
        )

    return scopes[scope]


# -------------------------------------------------
# Dataset-/Scope-Detail-Config laden (z.B. de_sample.yml, bayern.yml)
# -------------------------------------------------

def load_dataset_config_for_scope(scope: str) -> Dict[str, Any]:
    """
    Lädt die detaillierte Dataset-/Municipality-Config für einen Scope.

    Erwartet, dass in scopes.yml für den Scope ein Feld
      municipalities_config: "<dateiname>.yml"
    definiert ist.

    Unterstützt optional 'inherits_from' in der jeweiligen Datei:
      inherits_from: "bayern.yml"
    """
    scope_cfg = load_scope_config(scope)
    cfg_name = scope_cfg.get("municipalities_config")
    if not cfg_name:
        logger.info(
            "Keine 'municipalities_config' für Scope '%s' in scopes.yml definiert.",
            scope,
        )
        return {}

    configs_dir = project_root() / "configs"
    cfg_path = configs_dir / cfg_name

    # Basis-Config laden
    cfg = _load_yaml(cfg_path)
    if not cfg:
        return {}

    # Inheritance-Kette auflösen (z.B. de_bw.yml -> bayern.yml)
    seen: set[str] = set()
    while isinstance(cfg.get("inherits_from"), str):
        parent_name = cfg["inherits_from"]
        if parent_name in seen:
            logger.error(
                "Zyklische inherits_from-Referenz in Configs entdeckt: %s", parent_name
            )
            break
        seen.add(parent_name)
        parent_path = configs_dir / parent_name
        parent_cfg = _load_yaml(parent_path)
        if not parent_cfg:
            logger.error(
                "Eltern-Config '%s' (referenziert in '%s') konnte nicht geladen werden.",
                parent_name,
                cfg_name,
            )
            break

        # 'inherits_from' nur im Kind relevant, nicht nach oben propagieren
        child_cfg = dict(cfg)
        child_cfg.pop("inherits_from", None)

        cfg = _deep_merge(parent_cfg, child_cfg)

    return cfg


# -------------------------------------------------
# Struktur für alle relevanten Pfade
# -------------------------------------------------

@dataclass
class DatasetPaths:
    dataset_id: str
    sample_dataset_id: str

    # Haupt-Scope
    dataset_root_dir: Path
    staging_dir: Path
    extracted_dir: Path
    classified_dir: Path
    eval_dir: Path
    goldstandard_dir: Path
    meta_dir: Path

    # Sample-Scope (z.B. de_sample)
    sample_root_dir: Path
    sample_staging_dir: Path
    sample_extracted_dir: Path
    sample_classified_dir: Path
    sample_eval_dir: Path
    sample_goldstandard_dir: Path
    sample_meta_dir: Path


# -------------------------------------------------
# Ableitung aller benötigten Pfade aus scopes.yml
# -------------------------------------------------

def get_dataset_paths(scope_cfg: Dict[str, Any], scope_name: str) -> DatasetPaths:
    """
    Baut alle Pfade für einen Scope auf.

    Primäre Quelle ist scopes.yml:
      dataset_id, base_dir (optional), sample_dataset_id (optional)

    Falls für den Scope zusätzlich eine Detail-Config (z.B. de_sample.yml,
    bayern.yml, ...) mit einem Feld 'data_root' existiert, wird dieses als
    Basisverzeichnis verwendet. Andernfalls:

      dataset_root_dir = project_root()/ "data" / <dataset_id>
    """
    root = project_root()
    data_dir = root / "data"

    dataset_id = scope_cfg.get("dataset_id")
    if not dataset_id:
        raise KeyError(
            f"In der Scope-Konfiguration fehlt 'dataset_id' für Scope: {scope_name}"
        )

    # Sample-Dataset bestimmen:
    # 1) scopes.yml: sample_dataset_id
    # 2) sonst: <dataset_id>_sample (Fallback)
    sample_dataset_id = scope_cfg.get("sample_dataset_id", f"{dataset_id}_sample")

    # Detail-Config versuchen zu laden (z.B. de_sample.yml, bayern.yml)
    try:
        dataset_cfg = load_dataset_config_for_scope(scope_name)
    except Exception as exc:  # defensiv
        logger.error(
            "Fehler beim Laden der Dataset-Config für Scope '%s': %s",
            scope_name,
            exc,
        )
        dataset_cfg = {}

    # dataset_root_dir bestimmen:
    # 1) falls data_root in der Detail-Config gesetzt ist, verwende dieses (relativ zur Projektwurzel),
    # 2) sonst Standard: data/<dataset_id>
    data_root_override = dataset_cfg.get("data_root")
    if isinstance(data_root_override, str) and data_root_override.strip():
        dataset_root = root / data_root_override
    else:
        dataset_root = data_dir / dataset_id

    # Sample-Root analog; i.d.R. data/<sample_dataset_id>
    sample_root = data_dir / sample_dataset_id

    paths = DatasetPaths(
        dataset_id=dataset_id,
        sample_dataset_id=sample_dataset_id,
        dataset_root_dir=dataset_root,
        staging_dir=dataset_root / "staging",
        extracted_dir=dataset_root / "extracted",
        classified_dir=dataset_root / "classified",
        eval_dir=dataset_root / "eval",
        goldstandard_dir=dataset_root / "goldstandard",
        meta_dir=dataset_root / "meta",
        sample_root_dir=sample_root,
        sample_staging_dir=sample_root / "staging",
        sample_extracted_dir=sample_root / "extracted",
        sample_classified_dir=sample_root / "classified",
        sample_eval_dir=sample_root / "eval",
        sample_goldstandard_dir=sample_root / "goldstandard",
        sample_meta_dir=sample_root / "meta",
    )

    logger.debug(
        "DatasetPaths für Scope '%s': dataset_root_dir=%s, sample_root_dir=%s",
        scope_name,
        paths.dataset_root_dir,
        paths.sample_root_dir,
    )

    return paths


# -------------------------------------------------
# LLM-Konfiguration laden
# -------------------------------------------------

def load_llm_config() -> Dict[str, Any]:
    cfg_path = project_root() / "configs" / "llm.yml"
    if not cfg_path.exists():
        logger.info("Keine llm.yml gefunden, benutze Standardwerte.")
        return {}
    return _load_yaml(cfg_path)
