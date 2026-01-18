# generate_results_figures.py
# Usage (from repo root):
#   conda activate arlt
#   python generate_results_figures.py
#
# Outputs:
#   figures/fig_tradeoff_precision_coverage.pdf
#   figures/fig_tradeoff_precision_coverage.png
#   figures/fig_metrics_by_setting.pdf
#   figures/fig_metrics_by_setting.png
#   figures/table_metrics.csv
#   figures/table_metrics.tex

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------
# Configure your runs here
# -----------------------
RUNS = [
    {
        "key": "LLM few-shot (no post-gate)",
        "metrics": "data/de_by/goldstandard/eval_submission_mistral_few_full_gated_bg/llm_metrics.json",
    },
    {
        "key": "Post-gate pos_conf=0.80",
        "metrics": "data/de_by/goldstandard/eval_submission_mistral_few_full_gated_bg_post_pos0.80/llm_metrics_regated.json",
    },
    {
        "key": "Post-gate pos_conf=0.85 (selected)",
        "metrics": "data/de_by/goldstandard/eval_submission_mistral_few_full_gated_bg_post_pos0.85/llm_metrics_regated.json",
    },
    {
        "key": "Post-gate pos_conf=0.90",
        "metrics": "data/de_by/goldstandard/eval_submission_mistral_few_full_gated_bg_post_pos0.90/llm_metrics_regated.json",
    },
    {
        "key": "Post-gate pos_conf=0.95",
        "metrics": "data/de_by/goldstandard/eval_submission_mistral_few_full_gated_bg_post_pos0.95/llm_metrics_regated.json",
    },
]

OUTDIR = Path("figures")
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def build_table(runs: List[Dict[str, str]]) -> pd.DataFrame:
    rows = []
    for r in runs:
        m = load_json(r["metrics"])
        rows.append(
            {
                "setting": r["key"],
                "n_total": int(m.get("n_total", m.get("n", 0))),
                "n_classified": int(m.get("n_classified", m.get("n", 0))),
                "n_abstain": int(m.get("n_abstain", 0)),
                "coverage": safe_float(m.get("coverage", np.nan)),
                "tp": int(m.get("tp", 0)),
                "tn": int(m.get("tn", 0)),
                "fp": int(m.get("fp", 0)),
                "fn": int(m.get("fn", 0)),
                "precision": safe_float(m.get("precision", np.nan)),
                "recall_classified": safe_float(m.get("recall", np.nan)),
                "f1_classified": safe_float(m.get("f1", np.nan)),
                "recall_overall": safe_float(m.get("recall_overall", np.nan)),
                "accuracy_classified": safe_float(m.get("accuracy_classified", np.nan)),
            }
        )
    df = pd.DataFrame(rows)

    # For cases where precision/recall are 0.0 because no positives were predicted/classified,
    # keep them as-is (this is informative), but ensure numeric dtype.
    num_cols = [c for c in df.columns if c not in ("setting",)]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    return df


def export_tables(df: pd.DataFrame) -> None:
    df.to_csv(OUTDIR / "table_metrics.csv", index=False)

    # LaTeX (clean, minimal)
    latex_df = df.copy()
    latex_df["coverage"] = latex_df["coverage"].map(lambda v: f"{v:.2f}" if np.isfinite(v) else "")
    latex_df["precision"] = latex_df["precision"].map(lambda v: f"{v:.2f}" if np.isfinite(v) else "")
    latex_df["recall_classified"] = latex_df["recall_classified"].map(lambda v: f"{v:.2f}" if np.isfinite(v) else "")
    latex_df["recall_overall"] = latex_df["recall_overall"].map(lambda v: f"{v:.2f}" if np.isfinite(v) else "")
    latex_df["f1_classified"] = latex_df["f1_classified"].map(lambda v: f"{v:.2f}" if np.isfinite(v) else "")
    latex_df["accuracy_classified"] = latex_df["accuracy_classified"].map(lambda v: f"{v:.2f}" if np.isfinite(v) else "")

    cols = [
        "setting",
        "coverage",
        "precision",
        "recall_classified",
        "recall_overall",
        "fp",
        "tp",
        "n_abstain",
    ]
    latex = latex_df[cols].to_latex(
        index=False,
        escape=True,
        column_format="p{6cm}rrrrrrr",
        caption="Performance summary across LLM-only and post-gated operating points.",
        label="tab:perf_summary",
    )
    (OUTDIR / "table_metrics.tex").write_text(latex, encoding="utf-8")


def fig_precision_coverage(df: pd.DataFrame) -> None:
    x = df["coverage"].to_numpy(dtype=float)
    y = df["precision"].to_numpy(dtype=float)
    labels = df["setting"].tolist()

    plt.figure(figsize=(7.5, 4.8))
    plt.scatter(x, y)

    for xi, yi, lab in zip(x, y, labels):
        if not np.isfinite(xi) or not np.isfinite(yi):
            continue
        plt.annotate(
            lab,
            (xi, yi),
            textcoords="offset points",
            xytext=(6, 6),
            ha="left",
            fontsize=9,
        )

    plt.xlabel("Coverage (classified / total)")
    plt.ylabel("Precision (classified)")
    plt.title("Precision–Coverage Trade-off (LLM + Evidence Post-Gate)")
    plt.ylim(-0.05, 1.05)
    plt.xlim(max(0.0, np.nanmin(x) - 0.05), min(1.0, np.nanmax(x) + 0.05))
    plt.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(OUTDIR / "fig_tradeoff_precision_coverage.pdf")
    plt.savefig(OUTDIR / "fig_tradeoff_precision_coverage.png", dpi=200)
    plt.close()


def fig_metrics_bars(df: pd.DataFrame) -> None:
    # Bar chart: Coverage, Precision, Recall_overall (side by side, grouped)
    settings = df["setting"].tolist()
    cov = df["coverage"].to_numpy(dtype=float)
    prec = df["precision"].to_numpy(dtype=float)
    rec_overall = df["recall_overall"].to_numpy(dtype=float)

    # Replace NaN with 0 for plotting (still shown in tables)
    cov_p = np.nan_to_num(cov, nan=0.0)
    prec_p = np.nan_to_num(prec, nan=0.0)
    rec_p = np.nan_to_num(rec_overall, nan=0.0)

    idx = np.arange(len(settings))
    w = 0.25

    plt.figure(figsize=(10.5, 5.2))
    plt.bar(idx - w, cov_p, width=w, label="Coverage")
    plt.bar(idx, prec_p, width=w, label="Precision (classified)")
    plt.bar(idx + w, rec_p, width=w, label="Recall (overall)")

    # annotate bars
    def annotate(vals: np.ndarray, xoffs: float) -> None:
        for i, v in enumerate(vals):
            plt.text(
                idx[i] + xoffs,
                v + 0.02,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    annotate(cov_p, -w)
    annotate(prec_p, 0.0)
    annotate(rec_p, +w)

    plt.xticks(idx, [f"{i+1}" for i in range(len(settings))], fontsize=10)
    plt.xlabel("Setting index (see legend below)")
    plt.ylabel("Score")
    plt.title("Operating Points: Coverage vs Precision vs Overall Recall")
    plt.ylim(0.0, 1.15)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="upper right")

    # add mapping text below
    mapping = "\n".join([f"{i+1}: {s}" for i, s in enumerate(settings)])
    plt.gcf().text(0.01, -0.02, mapping, fontsize=9, va="top")

    plt.tight_layout()
    plt.savefig(OUTDIR / "fig_metrics_by_setting.pdf", bbox_inches="tight")
    plt.savefig(OUTDIR / "fig_metrics_by_setting.png", dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    df = build_table(RUNS)
    export_tables(df)

    fig_precision_coverage(df)
    fig_metrics_bars(df)

    print("Wrote:")
    print(f"  {OUTDIR / 'table_metrics.csv'}")
    print(f"  {OUTDIR / 'table_metrics.tex'}")
    print(f"  {OUTDIR / 'fig_tradeoff_precision_coverage.pdf'}")
    print(f"  {OUTDIR / 'fig_metrics_by_setting.pdf'}")


if __name__ == "__main__":
    main()
