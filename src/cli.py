# src/cli.py
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# WICHTIG:
# - Relative Imports funktionieren sauber mit: python -m src.cli ...
from .crawl.mvp import run_mvp
from .extraction.document_extractor import extract_documents
from .extraction.policy_extractor import build_policy_candidates
from .classification.llm_client import LLMClient

logger = logging.getLogger(__name__)


# -----------------------------
# Helpers
# -----------------------------
def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def require_llm_calls_enabled() -> None:
    if os.getenv("ALLOW_LLM_CALLS", "0") != "1":
        raise SystemExit("LLM-Aufrufe deaktiviert. Setze ALLOW_LLM_CALLS=1.")


# -----------------------------
# Commands
# -----------------------------
def cmd_crawl_mvp(args: argparse.Namespace) -> None:
    run_mvp(args.dataset, args.mvp_config, args.out)


def cmd_extract_documents(args: argparse.Namespace) -> None:
    raw_path = Path(args.raw)
    out_path = Path(args.out)
    ensure_parent(out_path)

    n = extract_documents(raw_path=raw_path, out_path=out_path)
    print(f"[extract-documents] wrote {n} -> {out_path}")


def cmd_extract_policies(args: argparse.Namespace) -> None:
    extracted_path = Path(args.extracted)
    out_path = Path(args.out)
    ensure_parent(out_path)

    keyword_cfg = Path(args.keyword_config) if args.keyword_config else None
    n = build_policy_candidates(
        extracted_path=extracted_path,
        out_path=out_path,
        keyword_config_path=keyword_cfg,
    )
    print(f"[extract-policies] wrote {n} -> {out_path}")


def cmd_auto_label_policies(args: argparse.Namespace) -> None:
    """
    Liest policy_candidates.ndjson (JSONL) und macht pro Candidate genau 1 LLM-Call.
    Unterstützt:
      --append: hängt an Output an
      --resume: skippt candidate_id, die schon im Output vorkommen (Output wird NICHT überschrieben)
    """
    require_llm_calls_enabled()

    candidates_path = Path(args.candidates)
    out_path = Path(args.out)
    ensure_parent(out_path)

    llm_cfg = load_yaml(args.llm_config)
    client = LLMClient(config=llm_cfg, model=args.model, profile=args.profile)

    written = 0
    processed = 0
    skipped_empty = 0
    skipped_existing = 0
    failed = 0

    max_n: Optional[int] = args.max

    # ------------------------------
    # Resume handling
    # ------------------------------
    processed_ids: set[str] = set()
    resume: bool = bool(getattr(args, "resume", False))
    append: bool = bool(getattr(args, "append", False))

    if resume and out_path.exists():
        try:
            with out_path.open("r", encoding="utf-8") as fprev:
                for ln in fprev:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        o = json.loads(ln)
                    except Exception:
                        continue
                    cid = o.get("candidate_id")
                    if isinstance(cid, str) and cid:
                        processed_ids.add(cid)
        except Exception as e:
            logger.warning("Could not read existing output for resume: %s", e)

    mode = "a" if (append or resume) else "w"

    with candidates_path.open("r", encoding="utf-8") as fin, out_path.open(mode, encoding="utf-8") as fout:
        for line in fin:
            if max_n is not None and processed >= max_n:
                break

            line = line.strip()
            if not line:
                continue

            c = json.loads(line)

            cid = c.get("candidate_id")
            if resume and isinstance(cid, str) and cid and cid in processed_ids:
                skipped_existing += 1
                continue

            processed += 1

            title = c.get("title")
            title_s = title.strip() if isinstance(title, str) else ""

            url = c.get("url") or c.get("source_url")
            url_s: str | None = url.strip() if isinstance(url, str) and url.strip() else None

            t = c.get("text") or c.get("snippet") or ""
            if not isinstance(t, str):
                t = ""
            t = t.strip()

            if not t:
                skipped_empty += 1
                continue

            if args.max_chars is not None and len(t) > args.max_chars:
                t = t[: args.max_chars]

            scope = c.get("municipality_id") or c.get("municipality")

            try:
                measures = client.extract_batch(texts=[t], titles=[title_s], urls=[url_s], scope=scope)
                m = measures[0] if measures and isinstance(measures[0], dict) else {}
            except Exception as e:
                failed += 1
                m = {
                    "label_source": "llm_error",
                    "confidence_score": 0.0,
                    "error": str(e),
                }

            m_out = dict(m)
            m_out["candidate_id"] = c.get("candidate_id") or c.get("id")
            m_out["document_id"] = c.get("document_id")
            m_out["municipality_id"] = scope
            m_out["url"] = url_s
            m_out["label_source"] = m_out.get("label_source") or "llm"
            m_out["llm_model"] = client.model_name
            m_out["llm_provider"] = client.provider_name
            m_out["llm_profile"] = client.profile_name

            fout.write(json.dumps(m_out, ensure_ascii=False) + "\n")
            written += 1

    print(
        f"[auto-label-policies] processed={processed} written={written} "
        f"skipped_empty={skipped_empty} skipped_existing={skipped_existing} failed={failed} -> {out_path}"
    )


def cmd_inspect_measures(args: argparse.Namespace) -> None:
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    key_fields = [
        "policy_area",
        "instrument_type",
        "instrument_subtype",
        "target_sector",
        "climate_dimension",
        "funding_program_level",
        "digitalization_level",
        "confidence_score",
    ]

    total = 0
    parse_errors = 0
    empty_lines = 0

    missing_counts: Counter[str] = Counter()
    value_counts: dict[str, Counter[str]] = {k: Counter() for k in key_fields if k != "confidence_score"}
    confidences: list[float] = []
    rows_for_sampling: list[dict[str, Any]] = []

    def norm_cat(v: Any) -> str:
        if v is None:
            return "null"
        if isinstance(v, str):
            s = v.strip()
            return s if s else "null"
        return str(v)

    def to_float(v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            try:
                return float(s)
            except Exception:
                return None
        return None

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                empty_lines += 1
                continue

            total += 1
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("row is not a dict")
            except Exception:
                parse_errors += 1
                continue

            for k in key_fields:
                if k not in row or row.get(k) in (None, "", []):
                    missing_counts[k] += 1

            for k, ctr in value_counts.items():
                ctr[norm_cat(row.get(k))] += 1

            c = to_float(row.get("confidence_score"))
            if c is not None:
                confidences.append(c)

            if args.keep_rows is None or len(rows_for_sampling) < args.keep_rows:
                rows_for_sampling.append(row)

    print(f"[inspect-measures] input={in_path}")
    print(f"  rows_total={total} empty_lines={empty_lines} parse_errors={parse_errors}")

    if total == 0:
        print("  (no rows)")
        return

    print("\nMissing rates (null/empty):")
    for k in key_fields:
        miss = missing_counts.get(k, 0)
        pct = (miss / total) * 100.0
        print(f"  {k:22s}  missing={miss:6d}  ({pct:5.1f}%)")

    print("\nTop values (categorical):")
    for k, ctr in value_counts.items():
        print(f"  {k}:")
        for val, cnt in ctr.most_common(args.top_k):
            pct = (cnt / total) * 100.0
            print(f"    {val:24s}  {cnt:6d}  ({pct:5.1f}%)")

    if confidences:
        buckets = [
            ("[0.00,0.20)", 0.00, 0.20),
            ("[0.20,0.40)", 0.20, 0.40),
            ("[0.40,0.60)", 0.40, 0.60),
            ("[0.60,0.80)", 0.60, 0.80),
            ("[0.80,1.00]", 0.80, 1.01),
        ]
        bctr = Counter()
        for c in confidences:
            for name, lo, hi in buckets:
                if lo <= c < hi:
                    bctr[name] += 1
                    break

        print("\nConfidence distribution:")
        nconf = len(confidences)
        for name, _, _ in buckets:
            cnt = bctr.get(name, 0)
            pct = (cnt / nconf) * 100.0 if nconf else 0.0
            print(f"  {name:12s}  {cnt:6d}  ({pct:5.1f}%)")

        conf_sorted = sorted(confidences)
        print(
            f"  stats: n={nconf}, min={min(confidences):.3f}, "
            f"p50={conf_sorted[nconf//2]:.3f}, max={max(confidences):.3f}"
        )
    else:
        print("\nConfidence distribution: (no numeric confidence_score values found)")

    if args.show_lowest and rows_for_sampling:
        def safe_conf(r: dict[str, Any]) -> float:
            v = to_float(r.get("confidence_score"))
            return v if v is not None else -1.0

        sorted_rows = sorted(rows_for_sampling, key=safe_conf)
        k = min(args.show_lowest, len(sorted_rows))
        print(f"\nLowest-confidence samples (k={k}):")
        for r in sorted_rows[:k]:
            out = {
                "candidate_id": r.get("candidate_id"),
                "municipality_id": r.get("municipality_id"),
                "confidence_score": r.get("confidence_score"),
                "policy_area": r.get("policy_area"),
                "instrument_type": r.get("instrument_type"),
                "target_sector": r.get("target_sector"),
                "measure_title": r.get("measure_title") or r.get("title"),
                "url": r.get("url") or r.get("source_url"),
            }
            print("  " + json.dumps(out, ensure_ascii=False))

    if args.show_random and rows_for_sampling:
        k = min(args.show_random, len(rows_for_sampling))
        print(f"\nRandom samples (k={k}):")
        for r in random.sample(rows_for_sampling, k=k):
            out = {
                "candidate_id": r.get("candidate_id"),
                "municipality_id": r.get("municipality_id"),
                "confidence_score": r.get("confidence_score"),
                "policy_area": r.get("policy_area"),
                "instrument_type": r.get("instrument_type"),
                "target_sector": r.get("target_sector"),
                "measure_title": r.get("measure_title") or r.get("title"),
                "url": r.get("url") or r.get("source_url"),
            }
            print("  " + json.dumps(out, ensure_ascii=False))


# -----------------------------
# Parser
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="k3-pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("crawl-mvp", help="MVP Crawl: HTML+PDF speichern (seeded, begrenzt)")
    p.add_argument("--dataset", required=True, help="z.B. configs/datasets/mvp_bayern.yaml")
    p.add_argument("--mvp-config", required=False, default=None, help="optional override YAML")
    p.add_argument("--out", required=True, help="z.B. data/de_by/staging/mvp")
    p.set_defaults(func=cmd_crawl_mvp)

    p = sub.add_parser("extract-documents", help="documents_raw.ndjson -> documents_extracted.ndjson")
    p.add_argument("--raw", required=True, help="Pfad zu documents_raw.ndjson")
    p.add_argument("--out", required=True, help="Pfad zu documents_extracted.ndjson")
    p.set_defaults(func=cmd_extract_documents)

    p = sub.add_parser("extract-policies", help="documents_extracted.ndjson -> policy_candidates.ndjson")
    p.add_argument("--extracted", required=True, help="Pfad zu documents_extracted.ndjson")
    p.add_argument("--out", required=True, help="Pfad zu policy_candidates.ndjson")
    p.add_argument("--keyword-config", required=False, default=None, help="optional keyword YAML")
    p.set_defaults(func=cmd_extract_policies)

    p = sub.add_parser("auto-label-policies", help="policy_candidates.ndjson -> measures_llm.ndjson (LLM)")
    p.add_argument("--candidates", required=True, help="Pfad zu policy_candidates.ndjson")
    p.add_argument("--out", required=True, help="Pfad zu measures_llm.ndjson")
    p.add_argument("--llm-config", required=True, help="z.B. configs/llm.yml")
    p.add_argument("--model", required=False, default=None, help="z.B. ollama:mistral")
    p.add_argument("--profile", required=False, default=None)
    p.add_argument("--max", type=int, default=None, help="max candidates")
    p.add_argument("--max-chars", type=int, default=6000)

    p.add_argument(
        "--append",
        action="store_true",
        help="Append to --out instead of overwriting (incremental runs).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="If --out exists, skip candidates whose candidate_id is already present in --out.",
    )
    p.set_defaults(func=cmd_auto_label_policies)

    p = sub.add_parser("inspect-measures", help="Quick sanity checks for measures_llm.ndjson (JSONL)")
    p.add_argument("--input", required=True, help="Pfad zu measures_llm.ndjson")
    p.add_argument("--top-k", type=int, default=10, help="Top-K category values to print")
    p.add_argument("--show-lowest", type=int, default=10, help="Print K lowest-confidence samples (0 disables)")
    p.add_argument("--show-random", type=int, default=10, help="Print K random samples (0 disables)")
    p.add_argument(
        "--keep-rows",
        type=int,
        default=5000,
        help="How many rows to keep in memory for sampling/sorting (bounded memory).",
    )
    p.set_defaults(func=cmd_inspect_measures)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
