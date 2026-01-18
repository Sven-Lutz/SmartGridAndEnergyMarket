import argparse
import json
import time
import re
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


SYSTEM_PROMPT = (
    "You are a STRICT classifier for municipal climate policy MEASURES.\n"
    "Definition: A MEASURE is an implemented, operational policy instrument.\n"
    "Necessary condition (must hold): TEXT contains explicit implementation evidence.\n"
    "Implementation evidence includes at least one of:\n"
    "- enacted/decided regulation/bylaw/requirement (e.g., Satzung, Verordnung, Pflicht)\n"
    "- approved budget/allocated funding with amounts or eligibility rules\n"
    "- launched program/scheme with concrete participation/rollout details\n"
    "- started/finished infrastructure project (procurement, construction, retrofit)\n"
    "- administrative action already taken (issued permits, inspections, enforcement)\n"
    "\n"
    "NOT a measure (label 0): goals, strategies, concepts, plans, monitoring intentions, announcements.\n"
    "If TEXT mentions programs/plans (e.g., 'Foerderprogramm', 'Waermeplanung') WITHOUT concrete rollout/budget/decision details: label = -1.\n"
    "\n"
    "Return ONLY JSON."
)

JSON_INSTRUCTIONS = (
    'Output JSON: {"label": 1|0|-1, "confidence": 0..1, "rationale": "..."} '
    "Use -1 if unsure. Keep rationale <= 20 words."
)

EVIDENCE_PATTERNS = [
    r"\b(beschluss|stadtratsbeschluss|gemeinderatsbeschluss|beschlossen)\b",
    r"\b(satzung|verordnung|richtlinie|pflicht|verpflichtend)\b",
    r"\b(gefördert|förderung|förderprogramm)\b.*\b(euro|€|million|tausend)\b",
    r"\b(mittel|budget|haushalt)\b.*\b(euro|€|million|tausend)\b",
    r"\b(ausschreibung|vergabe|beauftragt|auftrag)\b",
    r"\b(bau|gebaut|umbau|sanierung|modernisierung|errichtet|inbetriebnahme)\b",
    r"\b(umgesetzt|implementiert|in\s+kraft|inbetriebnahme)\b",
    r"\b(fördermittel|zuschuss|zuschüsse|bewilligt|bewilligung)\b",
    r"\b(stellt|stellt\s+.*\s+bereit|bereitgestellt)\b.*\b(mittel|budget|haushalt|förderung)\b",
    r"\b(beigetreten|mitglied|mitgliedschaft)\b.*\b(arbeitsgemeinschaft|netzwerk|verbund)\b",
]


HARD_MONEY = r"\b(euro|€|million|mio\.?|tausend)\b"
FUNDING_WORDS = r"\b(förder|förderung|förderprogramm|zuschuss|budget|haushalt|mittel)\b"
LEGAL_WORDS = r"\b(satzung|verordnung|richtlinie|pflicht|verpflichtend)\b"
DECISION_WORDS = r"\b(beschluss|stadtratsbeschluss|gemeinderatsbeschluss|beschlossen|verabschiedet)\b"
IMPLEMENT_WORDS = r"\b(ausschreibung|vergabe|beauftragt|auftrag|bau|gebaut|sanierung|modernisierung|errichtet|inbetriebnahme|eingeführt|umgesetzt)\b"

def has_implementation_evidence(text: str) -> bool:
    t = (text or "").lower()

    money_funding = re.search(HARD_MONEY, t) and re.search(FUNDING_WORDS, t)
    legal = re.search(LEGAL_WORDS, t) is not None
    decision_plus_impl = re.search(DECISION_WORDS, t) and (re.search(LEGAL_WORDS, t) or re.search(FUNDING_WORDS, t) or re.search(IMPLEMENT_WORDS, t))

    return bool(money_funding or legal or decision_plus_impl)



def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def ollama_chat(
    model: str,
    user_prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    host: str = "http://localhost:11434",
    timeout_s: int = 600,
) -> str:
    """
    Calls Ollama chat endpoint. Requires local ollama running.
    """
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]


def parse_json_response(text: str) -> Dict[str, Any]:
    """
    Robustly parse a JSON object from model output.
    """
    text = (text or "").strip()

    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError(f"Could not parse JSON from: {text[:200]}")


def build_fewshot_block(
    train_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    k_pos: int,
    k_neg: int,
    seed: int,
    max_chars: int,
) -> str:
    """
    Few-shot examples from TRAIN ONLY (no leakage).
    """
    rng = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    pos = rng[rng[label_col] == 1].head(k_pos)
    neg = rng[rng[label_col] == 0].head(k_neg)

    ex_lines: List[str] = []
    for _, row in pd.concat([pos, neg], ignore_index=True).iterrows():
        label = int(row[label_col])
        txt = str(row[text_col])
        if max_chars > 0:
            txt = txt[:max_chars]
        ex_lines.append(
            "EXAMPLE\n"
            f"TEXT:\n{txt}\n"
            f"OUTPUT:\n{{\"label\": {label}, \"confidence\": 0.9, \"rationale\": \"example\"}}\n"
        )
    return "\n".join(ex_lines).strip()


def make_prompt(text: str, fewshot_block: Optional[str]) -> str:
    parts = []
    if fewshot_block:
        parts.append("Labeled examples:\n" + fewshot_block)
    parts.append(JSON_INSTRUCTIONS)
    parts.append("TEXT:\n" + text)
    return "\n\n".join(parts)


def compute_metrics(pred_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Metrics are computed on classified-only rows; coverage reports abstention.
    pred_df columns:
      - gold_label (0/1)
      - pred_label (0/1 or NaN)
      - abstained (bool)
    """
    n_total = int(len(pred_df))
    n_abstain = int(pred_df["abstained"].sum())
    n_classified = n_total - n_abstain
    coverage = (n_classified / n_total) if n_total > 0 else 0.0

    eval_df = pred_df[~pred_df["abstained"]].copy()
    if len(eval_df) == 0:
        return {
            "n_total": n_total,
            "n_classified": 0,
            "n_abstain": n_abstain,
            "coverage": float(coverage),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "accuracy_classified": 0.0,
        }

    y_true = eval_df["gold_label"].to_numpy(dtype=int)
    y_pred = eval_df["pred_label"].to_numpy(dtype=int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
# overall recall: treat abstentions on positives as misses
    pos_total = int((pred_df["gold_label"] == 1).sum())
    tp_total = int(((pred_df["gold_label"] == 1) & (pred_df["pred_label"] == 1)).sum())
    recall_overall = (tp_total / pos_total) if pos_total > 0 else 0.0

    return {
        "n_total": n_total,
        "n_classified": int(n_classified),
        "n_abstain": int(n_abstain),
        "coverage": float(coverage),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy_classified": float((tp + tn) / len(eval_df)) if len(eval_df) > 0 else 0.0,
        "pos_total": pos_total,
        "tp_total": tp_total,
        "recall_overall": float(recall_overall),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_frac", type=float, default=0.30)

    parser.add_argument("--text_col", type=str, default="text_for_annotation")
    parser.add_argument("--label_col", type=str, default="gold_is_policy_measure")

    parser.add_argument("--model", type=str, default="mistral")
    parser.add_argument("--ollama_host", type=str, default="http://localhost:11434")

    parser.add_argument("--mode", type=str, choices=["zero", "few"], default="zero")
    parser.add_argument("--k_pos", type=int, default=3)
    parser.add_argument("--k_neg", type=int, default=3)

    parser.add_argument("--abstain_conf", type=float, default=0.6)
    parser.add_argument("--sleep_s", type=float, default=0.2)

    # Conservative runtime guards
    parser.add_argument("--timeout_s", type=int, default=600)
    parser.add_argument("--max_chars", type=int, default=2000)
    parser.add_argument("--max_test_n", type=int, default=10)

    parser.add_argument("--ensure_pos", action="store_true")
    parser.add_argument("--pos_conf", type=float, default=0.9)



    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.gold)
    for c in ["candidate_id", args.text_col, args.label_col]:
        assert c in df.columns, f"Missing column in gold CSV: {c}"

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_frac,
        random_state=args.seed,
        stratify=df[args.label_col],
    )

    fewshot_block = None
    if args.mode == "few":
        fewshot_block = build_fewshot_block(
            train_df=train_df,
            text_col=args.text_col,
            label_col=args.label_col,
            k_pos=args.k_pos,
            k_neg=args.k_neg,
            seed=args.seed,
            max_chars=args.max_chars,
        )

    test_iter = test_df.reset_index(drop=True)

    if args.max_test_n and args.max_test_n > 0:
        if args.ensure_pos:
            pos = test_iter[test_iter[args.label_col] == 1]
            neg = test_iter[test_iter[args.label_col] == 0]

            need_pos = min(len(pos), max(1, args.max_test_n // 4))
            need_neg = args.max_test_n - need_pos

            test_iter = pd.concat(
                [pos.head(need_pos), neg.head(need_neg)],
                ignore_index=True,
            )
        else:
            test_iter = test_iter.head(args.max_test_n)


    rows = []
    for _, row in test_iter.iterrows():
        cid = str(row["candidate_id"])
        text = str(row[args.text_col])
        if args.max_chars and args.max_chars > 0:
            text = text[: args.max_chars]
        gold_label = int(row[args.label_col])

        prompt = make_prompt(text=text, fewshot_block=fewshot_block)

        raw = ollama_chat(
            model=args.model,
            user_prompt=prompt,
            host=args.ollama_host,
            timeout_s=args.timeout_s,
        )

        try:
            parsed = parse_json_response(raw)
            label = int(parsed.get("label", -1))
            conf = float(parsed.get("confidence", 0.0))
            rationale = str(parsed.get("rationale", "")).strip()
        except Exception as e:
            label, conf, rationale = -1, 0.0, f"parse_error: {type(e).__name__}"




        abstained = (label == -1) or (conf < args.abstain_conf)
        pred_label: Optional[int] = None if abstained else (1 if label == 1 else 0)

        rows.append(
            {
                "candidate_id": cid,
                "gold_label": gold_label,
                "pred_label": pred_label if pred_label is not None else "",
                "abstained": bool(abstained),
                "confidence": conf,
                "rationale": rationale.replace("\n", " ").replace("\r", " "),
                "raw": raw.replace("\n", "\\n").replace("\r", "\\r"),
            }
        )

        if args.sleep_s and args.sleep_s > 0:
            time.sleep(args.sleep_s)

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(args.outdir / "llm_predictions.csv", index=False)

    metrics_input = pred_df.copy()
    metrics_input["pred_label"] = pd.to_numeric(metrics_input["pred_label"], errors="coerce")
    metrics_input["abstained"] = metrics_input["abstained"].astype(bool)
    metrics = compute_metrics(metrics_input)

    save_json(metrics, args.outdir / "llm_metrics.json")

    run_cfg = {
        "gold": str(args.gold),
        "outdir": str(args.outdir),
        "seed": args.seed,
        "test_frac": args.test_frac,
        "text_col": args.text_col,
        "label_col": args.label_col,
        "model": args.model,
        "ollama_host": args.ollama_host,
        "mode": args.mode,
        "k_pos": args.k_pos,
        "k_neg": args.k_neg,
        "abstain_conf": args.abstain_conf,
        "timeout_s": args.timeout_s,
        "max_chars": args.max_chars,
        "max_test_n": args.max_test_n,
        "sleep_s": args.sleep_s,
    }
    save_json(run_cfg, args.outdir / "run_config.json")


if __name__ == "__main__":
    main()
