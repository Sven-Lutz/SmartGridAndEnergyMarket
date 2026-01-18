import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


HARD_MONEY = r"\b(euro|€|million|mio\.?|tausend)\b"
FUNDING_WORDS = r"\b(förder|förderung|förderprogramm|zuschuss|budget|haushalt|mittel)\b"
LEGAL_WORDS = r"\b(satzung|verordnung|richtlinie|pflicht|verpflichtend)\b"
DECISION_WORDS = r"\b(beschluss|stadtratsbeschluss|gemeinderatsbeschluss|beschlossen|verabschiedet)\b"
IMPLEMENT_WORDS = r"\b(ausschreibung|vergabe|beauftragt|auftrag|bau|gebaut|sanierung|modernisierung|errichtet|inbetriebnahme|eingeführt|umgesetzt)\b"


def has_implementation_evidence(text: str) -> bool:
    t = (text or "").lower()
    money_funding = re.search(HARD_MONEY, t) and re.search(FUNDING_WORDS, t)
    legal = re.search(LEGAL_WORDS, t) is not None
    decision_plus_impl = re.search(DECISION_WORDS, t) and (
        re.search(LEGAL_WORDS, t) or re.search(FUNDING_WORDS, t) or re.search(IMPLEMENT_WORDS, t)
    )
    return bool(money_funding or legal or decision_plus_impl)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def compute_metrics(pred_df: pd.DataFrame) -> Dict[str, Any]:
    n_total = int(len(pred_df))
    n_abstain = int(pred_df["abstained"].sum())
    n_classified = n_total - n_abstain
    coverage = (n_classified / n_total) if n_total > 0 else 0.0

    eval_df = pred_df[~pred_df["abstained"]].copy()
    if len(eval_df) == 0:
        pos_total = int((pred_df["gold_label"] == 1).sum())
        tp_total = int(((pred_df["gold_label"] == 1) & (pred_df["pred_label"] == 1)).sum())
        recall_overall = (tp_total / pos_total) if pos_total > 0 else 0.0
        return {
            "n_total": n_total,
            "n_classified": 0,
            "n_abstain": n_abstain,
            "coverage": float(coverage),
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy_classified": 0.0,
            "pos_total": pos_total,
            "tp_total": tp_total,
            "recall_overall": float(recall_overall),
        }

    y_true = eval_df["gold_label"].to_numpy(dtype=int)
    y_pred = eval_df["pred_label"].to_numpy(dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

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
        "accuracy_classified": float((tp + tn) / len(eval_df)) if len(eval_df) else 0.0,
        "pos_total": pos_total,
        "tp_total": tp_total,
        "recall_overall": float(recall_overall),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", type=Path, required=True)
    ap.add_argument("--gold_csv", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--text_col", type=str, default="text_for_annotation")
    ap.add_argument("--pos_conf", type=float, default=0.9)
    ap.add_argument("--abstain_conf", type=float, default=0.6)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(args.pred_csv)
    gold = pd.read_csv(args.gold_csv)

    gold_small = gold[["candidate_id", args.text_col]].copy()
    gold_small["candidate_id"] = gold_small["candidate_id"].astype(str)

    pred["candidate_id"] = pred["candidate_id"].astype(str)
    merged = pred.merge(gold_small, on="candidate_id", how="left")
    assert merged[args.text_col].notna().all(), "Some candidate_id not found in gold CSV"

    merged["gold_label"] = merged["gold_label"].astype(int)
    merged["confidence"] = pd.to_numeric(merged["confidence"], errors="coerce").fillna(0.0)
    merged["pred_label"] = pd.to_numeric(merged["pred_label"], errors="coerce")
    merged["abstained"] = merged["abstained"].astype(bool)

    new_abstained = []
    new_pred_label = []

    for _, r in merged.iterrows():
        conf = float(r["confidence"])
        text = str(r[args.text_col])

        abst = bool(r["abstained"])
        pl = r["pred_label"]
        pl_int = None if pd.isna(pl) else int(pl)

        evidence = has_implementation_evidence(text)

        # stricter: accept positive only if evidence AND high confidence
        if (pl_int == 1) and (not evidence or conf < args.pos_conf):
            abst = True
            pl_int = None
            conf = min(conf, 0.49)

        # global abstain threshold
        if (not abst) and (conf < args.abstain_conf):
            abst = True
            pl_int = None

        new_abstained.append(bool(abst))
        new_pred_label.append("" if pl_int is None else pl_int)

    merged["abstained"] = new_abstained
    merged["pred_label"] = new_pred_label

    out_pred = merged[pred.columns.tolist()].copy()
    out_pred.to_csv(args.outdir / "llm_predictions_regated.csv", index=False)

    met_df = merged.copy()
    met_df["pred_label"] = pd.to_numeric(met_df["pred_label"], errors="coerce")
    metrics = compute_metrics(met_df[["gold_label", "pred_label", "abstained"]])

    save_json(metrics, args.outdir / "llm_metrics_regated.json")


if __name__ == "__main__":
    main()
