# results_plot.py
# Publication-ready plots for Section 5 (Results)
# - Confusion matrix comparison (baseline vs bootstrapped)
# - Metrics comparison (Precision/Recall/F1/Accuracy)
# - Optional Precision/Recall-only plot
#
# Usage:
#   python results_plot.py --outdir results/figures
#   python results_plot.py --log runs/<RUN_ID>/stdout.txt --outdir results/figures
#
# Notes:
# - If --log is provided, the script parses dict-literals from lines like:
#     Baseline (holdout): {'n': 45, 'tp': 2, 'tn': 41, 'fp': 1, 'fn': 1, ...}
#     Bootstrapped (holdout): {...}
# - It writes BOTH .png and .pdf for easy LaTeX inclusion.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Mapping, cast
import argparse
import re
import ast

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns


# ----------------------------
# Theme (type-safe)
# ----------------------------
@dataclass(frozen=True)
class Theme:
    dpi: int = 300
    font_family: str = "DejaVu Sans"
    title_size: int = 18
    label_size: int = 12
    tick_size: int = 11
    legend_size: int = 11
    colors: Mapping[str, str] = field(
        default_factory=lambda: {
            "baseline": "#0072B2",
            "boot": "#E69F00",
            "grid": "#D0D0D0",
            "text": "#111111",
        }
    )


def apply_theme(theme: Theme) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": theme.dpi,
            "savefig.dpi": theme.dpi,
            "font.family": theme.font_family,
            "axes.titlesize": theme.title_size,
            "axes.labelsize": theme.label_size,
            "xtick.labelsize": theme.tick_size,
            "ytick.labelsize": theme.tick_size,
            "legend.fontsize": theme.legend_size,
            "axes.grid": True,
            "grid.color": theme.colors["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# ----------------------------
# Metrics helpers
# ----------------------------
@dataclass(frozen=True)
class Metrics:
    precision: float
    recall: float
    f1: float
    accuracy: float


def ensure_outdir(outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def save_fig(fig: Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def cm_from_counts(d: Mapping[str, Any]) -> np.ndarray:
    """
    Expected keys: tp, tn, fp, fn
    Returns cm:
        rows = gold (true), cols = predicted
        [[TN, FP],
         [FN, TP]]
    """
    tp = int(d.get("tp", 0))
    tn = int(d.get("tn", 0))
    fp = int(d.get("fp", 0))
    fn = int(d.get("fn", 0))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def metrics_from_cm(cm: np.ndarray) -> Metrics:
    cmf = cm.astype(float)
    tn, fp = cmf[0, 0], cmf[0, 1]
    fn, tp = cmf[1, 0], cmf[1, 1]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return Metrics(precision, recall, f1, accuracy)


def extract_dict_after_key(log_text: str, key: str) -> Optional[Dict[str, Any]]:
    """
    Extract a python dict literal following a key in logs, e.g.
      "Baseline (holdout): {'n': 45, 'tp': 2, ...}"
    """
    m = re.search(rf"{re.escape(key)}\s*(\{{.*\}})", log_text)
    if not m:
        return None
    return cast(Dict[str, Any], ast.literal_eval(m.group(1)))


def load_cm_from_log(log_path: Path, key: str) -> np.ndarray:
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    d = extract_dict_after_key(txt, key)
    if d is None:
        raise ValueError(f"Key not found in log: {key}")
    return cm_from_counts(d)


# ----------------------------
# Plotting
# ----------------------------
def plot_cm(ax: Axes, cm: np.ndarray, title: str, labels: Tuple[str, str]) -> None:
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cbar=False,
        square=True,
        ax=ax,
        xticklabels=list(labels),
        yticklabels=list(labels),
        annot_kws={"size": 16},
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(title, pad=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)


def plot_confusion_matrix_comparison(
    cm_baseline: np.ndarray,
    cm_boot: np.ndarray,
    out_base: Path,
    labels: Tuple[str, str] = ("No Measure", "Measure"),
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    ax0 = cast(Axes, axes[0])
    ax1 = cast(Axes, axes[1])
    plot_cm(ax0, cm_baseline, "Baseline (gold only)", labels)
    plot_cm(ax1, cm_boot, "Bootstrapped (gold + weak)", labels)
    fig.tight_layout()
    save_fig(fig, out_base)


def plot_metrics_comparison(
    baseline: Metrics,
    boot: Metrics,
    out_base: Path,
    title: str = "Metrics Comparison on Gold Test Set",
) -> None:
    names = ["Precision", "Recall", "F1", "Accuracy"]
    base_vals = np.array([baseline.precision, baseline.recall, baseline.f1, baseline.accuracy], dtype=float)
    boot_vals = np.array([boot.precision, boot.recall, boot.f1, boot.accuracy], dtype=float)

    xs = np.arange(len(names), dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax = cast(Axes, ax)
    ax.bar(xs - width / 2, base_vals, width, label="Baseline")
    ax.bar(xs + width / 2, boot_vals, width, label="Bootstrapped")

    ax.set_title(title, pad=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, out_base)


def plot_precision_recall_only(
    baseline: Metrics,
    boot: Metrics,
    out_base: Path,
    title: str = "Precision and Recall Comparison",
) -> None:
    names = ["Precision", "Recall"]
    base_vals = np.array([baseline.precision, baseline.recall], dtype=float)
    boot_vals = np.array([boot.precision, boot.recall], dtype=float)

    xs = np.arange(len(names), dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax = cast(Axes, ax)
    ax.bar(xs - width / 2, base_vals, width, label="Baseline")
    ax.bar(xs + width / 2, boot_vals, width, label="Bootstrapped")

    ax.set_title(title, pad=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, out_base)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="results/figures", help="Output directory for plots")
    parser.add_argument("--log", type=str, default="", help="Path to run log (stdout.txt) to load metrics from")
    parser.add_argument("--baseline_key", type=str, default="Baseline (holdout):", help="Log key for baseline dict")
    parser.add_argument("--boot_key", type=str, default="Bootstrapped (holdout):", help="Log key for boot dict")
    args = parser.parse_args()

    theme = Theme()
    apply_theme(theme)

    outdir = ensure_outdir(Path(args.outdir))

    # --- Load confusion matrices ---
    if args.log:
        log_path = Path(args.log)
        cm_baseline = load_cm_from_log(log_path, args.baseline_key)
        cm_boot = load_cm_from_log(log_path, args.boot_key)
    else:
        # Fallback example baseline from your earlier snippet
        baseline_dict = {"n": 45, "tp": 2, "tn": 41, "fp": 1, "fn": 1}
        cm_baseline = cm_from_counts(baseline_dict)

        # Placeholder bootstrapped (replace or use --log for real numbers)
        boot_dict = {"n": 45, "tp": 2, "tn": 42, "fp": 0, "fn": 1}
        cm_boot = cm_from_counts(boot_dict)

    m_base = metrics_from_cm(cm_baseline)
    m_boot = metrics_from_cm(cm_boot)

    # --- Plots ---
    plot_confusion_matrix_comparison(
        cm_baseline,
        cm_boot,
        out_base=outdir / "cm_comparison",
        labels=("No Measure", "Measure"),
    )

    plot_metrics_comparison(
        m_base,
        m_boot,
        out_base=outdir / "metrics_comparison",
        title="Metrics Comparison on Gold Test Set",
    )

    plot_precision_recall_only(
        m_base,
        m_boot,
        out_base=outdir / "precision_recall_comparison",
        title="Precision and Recall Comparison",
    )

    print("Saved plots to:", outdir.resolve())
    print("Baseline CM:\n", cm_baseline)
    print("Bootstrapped CM:\n", cm_boot)
    print("Baseline metrics:", m_base)
    print("Bootstrapped metrics:", m_boot)


if __name__ == "__main__":
    main()
