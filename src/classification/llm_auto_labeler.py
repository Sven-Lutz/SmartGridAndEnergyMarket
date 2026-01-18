# src/classification/llm_auto_labeler.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.utils.logging_config import get_logger
from src.classification.llm_client import LLMClient

logger = get_logger(__name__)


@dataclass(frozen=True)
class LLMAutoLabelConfig:
    """
    MVP-Config:
    - batch_size: wie viele Kandidaten pro LLM-Batch (extract_batch verarbeitet ohnehin einzeln,
      aber wir bündeln die Input-Listen)
    - max_items: Top-N Kandidaten (Kostenkontrolle)
    - max_chars: harte Textlänge pro Kandidat (Kostenkontrolle)
    """
    model: str | None = None
    profile: str | None = None
    batch_size: int = 5
    max_items: Optional[int] = 30
    max_chars: int = 6000  # ~ wenige tausend Tokens, je nach Sprache/Format


def _require_llm_calls_enabled() -> None:
    """
    Safety Switch gegen versehentliche Kosten.
    """
    if os.getenv("ALLOW_LLM_CALLS", "0") != "1":
        raise SystemExit("LLM-Aufrufe deaktiviert. Setze ALLOW_LLM_CALLS=1.")


def _iter_ndjson(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"NDJSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _cap_text(text: str | None, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars]


def _pick_text_for_llm(rec: Dict[str, Any], max_chars: int) -> str:
    """
    Kandidaten sollen idealerweise schon einen snippet/text enthalten.
    Wir unterstützen mehrere Feldnamen, damit es in eure Pipeline passt.
    """
    for key in ("text", "snippet", "text_snippet", "evidence_snippet", "content"):
        val = rec.get(key)
        if isinstance(val, str) and val.strip():
            return _cap_text(val, max_chars)
    return ""


def _pick_title(rec: Dict[str, Any]) -> str:
    val = rec.get("title")
    return val.strip() if isinstance(val, str) else ""


def _pick_url(rec: Dict[str, Any]) -> str | None:
    for key in ("url", "source_url"):
        val = rec.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _pick_scope(rec: Dict[str, Any]) -> str | None:
    # LLMClient nutzt "scope" eher als Herkunftsangabe; bei euch ist municipality_id ok.
    for key in ("municipality_id", "municipality", "scope"):
        val = rec.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _iter_batches(items: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), batch_size):
        yield list(items[i:i + batch_size])


def auto_label_candidates_with_llm(
    *,
    candidates_path: Path,
    out_path: Path,
    llm_config: Dict[str, Any],
    cfg: LLMAutoLabelConfig,
) -> int:
    """
    Liest policy_candidates.ndjson und ruft LLMClient.extract_batch(...) auf.
    Schreibt measures_llm.ndjson (1 Measure-Dict pro Candidate; gleiche Reihenfolge).

    Erwartung: candidates sind bereits stark vorgefiltert (MVP).
    """
    _require_llm_calls_enabled()

    candidates = list(_iter_ndjson(candidates_path))
    if cfg.max_items is not None:
        candidates = candidates[: cfg.max_items]

    if not candidates:
        logger.warning("Keine Kandidaten in %s", candidates_path)
        return 0

    client = LLMClient(config=llm_config, model=cfg.model, profile=cfg.profile)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        for batch in _iter_batches(candidates, cfg.batch_size):
            texts: List[str] = []
            titles: List[str] = []
            urls: List[str | None] = []
            scopes: List[str | None] = []

            for rec in batch:
                texts.append(_pick_text_for_llm(rec, cfg.max_chars))
                titles.append(_pick_title(rec))
                urls.append(_pick_url(rec))
                scopes.append(_pick_scope(rec))

            # extract_batch nimmt einen einzelnen scope-Wert, nicht pro item.
            # MVP: wir setzen scope=None und verlassen uns auf source_url/title.
            # Alternativ: wenn alle im Batch gleiche municipality_id haben, könnten wir sie setzen.
            # Für Robustheit: scope None.
            measures = client.extract_batch(
                texts=texts,
                titles=titles,
                urls=urls,
                scope=None,
            )

            # Robustheit: gleiche Länge erwarten
            if len(measures) != len(batch):
                logger.warning(
                    "LLMClient.extract_batch returned %d results for %d inputs. "
                    "Wir alignen per Index (min length).",
                    len(measures),
                    len(batch),
                )

            n = min(len(measures), len(batch))
            for i in range(n):
                rec = batch[i]
                m = measures[i] if isinstance(measures[i], dict) else {}

                # Candidate-Metadaten hinzufügen (für Traceability)
                m_out = dict(m)
                m_out["candidate_id"] = rec.get("candidate_id") or rec.get("id")
                m_out["document_id"] = rec.get("document_id")
                m_out["municipality_id"] = rec.get("municipality_id") or rec.get("municipality")
                m_out["url"] = _pick_url(rec)
                m_out["title"] = m_out.get("title") or _pick_title(rec)

                # Optional: Kennzeichnung für MVP
                m_out["label_source"] = m_out.get("label_source") or "llm"
                m_out["llm_model"] = client.model_name
                m_out["llm_provider"] = client.provider_name
                m_out["llm_profile"] = client.profile_name

                f_out.write(json.dumps(m_out, ensure_ascii=False) + "\n")
                written += 1

    logger.info("LLM-Auto-Labeling fertig: %d Zeilen -> %s", written, out_path)
    return written
