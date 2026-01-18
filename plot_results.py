from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Sequence
import re
import ast

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns


# ----------------------------
# Config
# ----------------------------
DEFAULT_OUTDIR = Path("results/figures")  # recommended: commit or keep under results/


# ----------------------------
# Data structures
# ----------------------------
@dataclass(frozen=True)
class Metrics:
    precision: float
    recall: float
    f1: float
    accuracy: float


# ----------------------------
# Helpers
# ----------------------------
def ensure_outdir(outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def save(fig: Figure, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def metrics_from_cm(cm: np.ndarray) -> Metrics:
    """
    cm layout:
        rows = gold (true), cols = predicted
        [[TN, FP],
         [FN, TP]]
    """
    cm = np.asarray(cm, dtype=float)
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return Metrics(precision=precision, recall=recall, f1=f1, accuracy=accuracy)


def cm_from_dict(d: dict) -> np.ndarray:
    """
    Expected keys: tp, tn, fp, fn
    """
    tp = int(d.get("tp", 0))
    tn = int(d.get("tn", 0))
    fp = int(d.get("fp", 0))
    fn = int(d.get("fn", 0))
    return np.array([[tn, fp],
                     [fn, tp]], dtype=int)


def extract_metrics_dict(log_text: str, key: str) -> dict | None:
    """
    Extracts a python-dict literal following a key in logs, e.g.
    "Baseline (holdout): {'n': 45, 'tp': 2, ...}"
    """
    m = re.search(rf"{re.escape(key)}\s*(\{{.*\}})", log_text)
    if not m:
        return None
    return ast.literal_eval(m.group(1))


def load_cm_from_log(log_path: Path, key: str) -> np.ndarray:
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    d = extract_metrics_dict(txt, key)
    if d is None:
        raise ValueError(f"Key not found in log: {key}")
    return cm_from_dict(d)


# ----------------------------
# Plot functions
# ----------------------------
def plot_confusion_matrix(
    cm: np.ndarray,
    title: str,
    out_base: Path,
    labels: tuple[str, str] = ("No Measure", "Measure"),
    dpi: int = 300,
) -> None:
    """
    Saves both PNG and PDF:
      <out_base>.png
      <out_base>.pdf
    """
    cm = np.asarray(cm, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".0f",
        cbar=False,
        square=True,
        ax=ax,
        xticklabels=list(labels),
        yticklabels=list(labels),
        annot_kws={"size": 22},
    )

    ax.set_title(title, fontsize=22, pad=16)
    ax.set_xlabel("Predicted", fontsize=16, labelpad=10)
    ax.set_ylabel("Gold", fontsize=16, labelpad=10)
    ax.tick_params(axis="x", labelsize=16, rotation=0)
    ax.tick_params(axis="y", labelsize=16, rotation=0)

    fig.tight_layout()
    save(fig, out_base.with_suffix(".png"))

    # recreate for PDF (avoids backend quirks)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".0f",
        cbar=False,
        square=True,
        ax=ax,
        xticklabels=list(labels),
        yticklabels=list(labels),
        annot_kws={"size": 22},
    )
    ax.set_title(title, fontsize=22, pad=16)
    ax.set_xlabel("Predicted", fontsize=16, labelpad=10)
    ax.set_ylabel("Gold", fontsize=16, labelpad=10)
    ax.tick_params(axis="x", labelsize=16, rotation=0)
    ax.tick_params(axis="y", labelsize=16, rotation=0)
    fig.tight_layout()
    save(fig, out_base.with_suffix(".pdf"))


def plot_precision_recall_bars(
    names: Sequence[str],
    precisions: Sequence[float],
    recalls: Sequence[float],
    title: str,
    out_base: Path,
    dpi: int = 300,
) -> None:
    """
    Saves both PNG and PDF:
      <out_base>.png
      <out_base>.pdf
    """
    xs = np.arange(len(names), dtype=float)
    prec_arr = np.asarray(list(precisions), dtype=float)
    rec_arr = np.asarray(list(recalls), dtype=float)
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=dpi)
    ax.bar(xs - width / 2, prec_arr, width, label="Precision")
    ax.bar(xs + width / 2, rec_arr, width, label="Recall")

    ax.set_title(title, fontsize=20, pad=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(names), fontsize=14)
    ax.legend(fontsize=12)

    fig.tight_layout()
    save(fig, out_base.with_suffix(".png"))

    # PDF
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=dpi)
    ax.bar(xs - width / 2, prec_arr, width, label="Precision")
    ax.bar(xs + width / 2, rec_arr, width, label="Recall")
    ax.set_title(title, fontsize=20, pad=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(names), fontsize=14)
    ax.legend(fontsize=12)
    fig.tight_layout()
    save(fig, out_base.with_suffix(".pdf"))


def plot_metrics_comparison(
    baseline: Metrics,
    bootstrapped: Metrics,
    out_base: Path,
    title: str = "Metrics Comparison (Baseline vs Bootstrapped)",
    dpi: int = 300,
) -> None:
    """
    Bar chart for Precision/Recall/F1/Accuracy, saves PNG+PDF.
    """
    metrics_names = ["Precision", "Recall", "F1", "Accuracy"]
    base_vals = np.array([baseline.precision, baseline.recall, baseline.f1, baseline.accuracy], dtype=float)
    boot_vals = np.array([bootstrapped.precision, bootstrapped.recall, bootstrapped.f1, bootstrapped.accuracy], dtype=float)

    xs = np.arange(len(metrics_names), dtype=float)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=dpi)
    ax.bar(xs - width / 2, base_vals, width, label="Baseline")
    ax.bar(xs + width / 2, boot_vals, width, label="Bootstrapped")

    ax.set_title(title, fontsize=18, pad=12)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(metrics_names, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(fontsize=11)

    fig.tight_layout()
    save(fig, out_base.with_suffix(".png"))

    # PDF
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=dpi)
    ax.bar(xs - width / 2, base_vals, width, label="Baseline")
    ax.bar(xs + width / 2, boot_vals, width, label="Bootstrapped")
    ax.set_title(title, fontsize=18, pad=12)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(metrics_names, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(fontsize=11)
    fig.tight_layout()
    save(fig, out_base.with_suffix(".pdf"))


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    outdir = ensure_outdir(DEFAULT_OUTDIR)

    # --- Option 1: hard-coded example numbers (from your screenshots) ---
    cm_classifier = np.array([[140, 1],
                              [3,   6]], dtype=int)

    cm_llm_gate = np.array([[110, 31],
                            [7,   2]], dtype=int)

    # Save confusion matrices
    plot_confusion_matrix(
        cm_classifier,
        title="Supervised Classifier – Confusion Matrix",
        out_base=outdir / "cm_classifier",
    )
    plot_confusion_matrix(
        cm_llm_gate,
        title="LLM Gate – Confusion Matrix",
        out_base=outdir / "cm_llm_gate",
    )

    # Save PR comparison
    m_clf = metrics_from_cm(cm_classifier)
    m_gate = metrics_from_cm(cm_llm_gate)

    plot_precision_recall_bars(
        names=["LLM Gate", "Classifier"],
        precisions=[m_gate.precision, m_clf.precision],
        recalls=[m_gate.recall, m_clf.recall],
        title="Measure Detection Performance",
        out_base=outdir / "precision_recall_comparison",
    )

    # --- Option 2: load from logs (uncomment + adjust paths/keys) ---
    # log_path = Path("runs/<YOUR_RUN>/stdout.txt")
    # cm_baseline = load_cm_from_log(log_path, "Baseline (holdout):")
    # cm_boot = load_cm_from_log(log_path, "Bootstrapped (holdout):")
    #
    # plot_confusion_matrix(cm_baseline, "Baseline – Confusion Matrix", outdir / "cm_baseline")
    # plot_confusion_matrix(cm_boot, "Bootstrapped – Confusion Matrix", outdir / "cm_bootstrapped")
    #
    # m_base = metrics_from_cm(cm_baseline)
    # m_boot = metrics_from_cm(cm_boot)
    # plot_metrics_comparison(m_base, m_boot, outdir / "metrics_comparison")

    print(f"Saved figures to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
