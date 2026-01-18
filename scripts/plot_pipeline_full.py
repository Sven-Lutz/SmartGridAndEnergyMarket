# scripts/plot_pipeline_full.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure


# ============================================================
# Theme (consistent across all plots)
# ============================================================
@dataclass(frozen=True)
class PlotTheme:
    colors: Dict[str, str]
    dpi: int = 300
    font_scale: float = 1.1


def default_theme() -> PlotTheme:
    # Okabe-Ito palette (color-blind safe, print-friendly)
    return PlotTheme(colors={
        "baseline": "#0072B2",
        "bootstrapped": "#D55E00",
        "rules": "#009E73",
        "llm": "#CC79A7",
        "hybrid": "#E69F00",
        "pos": "#009E73",
        "neg": "#D55E00",
        "abstain": "#999999",
        "neutral": "#999999",
    })


def apply_theme(theme: PlotTheme) -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font_scale=theme.font_scale,
        rc={
            "figure.dpi": theme.dpi,
            "savefig.dpi": theme.dpi,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )


# ============================================================
# Helpers
# ============================================================
def ensure_outdir(outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def save(fig: Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def read_csv_auto(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")

    if df.shape[1] == 1:
        col0 = str(df.columns[0])
        if ";" in col0:
            df = pd.read_csv(path, sep=";", engine="python", on_bad_lines="skip")
    return df


def normalize_binary_label(series: pd.Series) -> pd.Series:
    s = series.copy()
    if s.dtype == bool:
        return s.astype("float").astype("Int64")

    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().any():
        return s_num.round().astype("Int64")

    s_str = s.astype(str).str.strip().str.lower()
    mapping = {"true": 1, "false": 0, "yes": 1, "no": 0, "y": 1, "n": 0}
    out = s_str.map(mapping)
    return out.astype("Int64")


def get_text_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["text_for_annotation", "text", "content", "paragraph", "snippet"]:
        if c in df.columns:
            return c
    return None


def get_id_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["id", "doc_id", "document_id", "source_id"]:
        if c in df.columns:
            return c
    return None


def get_muni_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["municipality_id", "municipality", "muni", "city_id", "city"]:
        if c in df.columns:
            return c
    return None


# ============================================================
# Plots (English, fully labeled)
# ============================================================
def plot_dataset_sizes(items: Sequence[Tuple[str, int]], out_base: Path, theme: PlotTheme) -> None:
    names = [k for k, _ in items]
    vals = np.array([v for _, v in items], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=theme.dpi)
    ax.bar(np.arange(len(names)), vals, color=theme.colors["neutral"])
    ax.set_title("Dataset sizes across pipeline artifacts")
    ax.set_ylabel("Number of rows")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=0)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=10)
    save(fig, out_base)


def plot_label_balance(rows: Sequence[Tuple[str, int, int, int]], out_base: Path, theme: PlotTheme) -> None:
    names = [r[0] for r in rows]
    pos = np.array([r[1] for r in rows], dtype=float)
    neg = np.array([r[2] for r in rows], dtype=float)
    na = np.array([r[3] for r in rows], dtype=float)

    x = np.arange(len(names), dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=theme.dpi)

    ax.bar(x, neg, label="Negative", color=theme.colors["neg"])
    ax.bar(x, pos, bottom=neg, label="Positive", color=theme.colors["pos"])
    ax.bar(x, na, bottom=neg + pos, label="Unlabeled / NA", color=theme.colors["abstain"])

    ax.set_title("Label distribution across pipeline stages")
    ax.set_ylabel("Number of rows")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=0)
    ax.legend()

    totals = neg + pos + na
    for i, t in enumerate(totals):
        ax.text(i, t, f"{int(t)}", ha="center", va="bottom", fontsize=10)

    save(fig, out_base)


def plot_text_length_distribution(
    series_list: Sequence[pd.Series],
    labels: Sequence[str],
    out_base: Path,
    theme: PlotTheme,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=theme.dpi)
    for s, name in zip(series_list, labels):
        lens = s.fillna("").astype(str).str.len().to_numpy()
        ax.hist(lens, bins=40, alpha=0.5, label=name)
    ax.set_title("Text length distribution (characters)")
    ax.set_xlabel("Text length (characters)")
    ax.set_ylabel("Count")
    ax.legend()
    save(fig, out_base)


def plot_weak_rule_coverage(weak_df: pd.DataFrame, out_base: Path, theme: PlotTheme) -> None:
    if "rule_label" not in weak_df.columns:
        return
    s = weak_df["rule_label"]
    pos = int((s == 1).sum())
    neg = int((s == 0).sum())
    abst = int(s.isna().sum())

    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=theme.dpi)
    ax.bar(["Negative", "Positive", "Abstain"], [neg, pos, abst],
           color=[theme.colors["neg"], theme.colors["pos"], theme.colors["abstain"]])
    ax.set_title("Weak rule labeling coverage")
    ax.set_ylabel("Number of documents")
    for i, v in enumerate([neg, pos, abst]):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=10)
    save(fig, out_base)


def plot_rule_score_distribution(weak_df: pd.DataFrame, out_base: Path, theme: PlotTheme) -> None:
    if "rule_score" not in weak_df.columns:
        return
    df = weak_df.copy()
    df["rule_score"] = pd.to_numeric(df["rule_score"], errors="coerce")
    df = df[df["rule_score"].notna()]

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=theme.dpi)

    if "rule_label" in df.columns:
        pos = df[df["rule_label"] == 1]["rule_score"].to_numpy()
        neg = df[df["rule_label"] == 0]["rule_score"].to_numpy()
        abst = df[df["rule_label"].isna()]["rule_score"].to_numpy()
        if len(pos) > 0:
            ax.hist(pos, bins=30, alpha=0.55, label="Positive", color=theme.colors["pos"])
        if len(neg) > 0:
            ax.hist(neg, bins=30, alpha=0.55, label="Negative", color=theme.colors["neg"])
        if len(abst) > 0:
            ax.hist(abst, bins=30, alpha=0.35, label="Abstain", color=theme.colors["abstain"])
    else:
        ax.hist(df["rule_score"].to_numpy(), bins=30, alpha=0.7, color=theme.colors["rules"])

    ax.set_title("Rule score distribution")
    ax.set_xlabel("Rule score")
    ax.set_ylabel("Count")
    ax.legend()
    save(fig, out_base)


def extract_rule_frequencies(weak_df: pd.DataFrame, col: str = "matched_rules", sep: str = ";") -> pd.Series:
    if col not in weak_df.columns:
        return pd.Series(dtype=int)
    s = weak_df[col].dropna().astype(str)
    tokens: list[str] = []
    for v in s:
        parts = [p.strip() for p in v.split(sep) if p.strip()]
        tokens.extend(parts)
    if not tokens:
        return pd.Series(dtype=int)
    return pd.Series(tokens).value_counts()


def plot_top_rules(freqs: pd.Series, title: str, color: str, out_base: Path, theme: PlotTheme) -> None:
    if freqs.empty:
        return
    freqs = freqs.head(15)

    y = list(freqs.index[::-1])
    width = np.asarray(freqs.to_numpy(), dtype=float)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6.0), dpi=theme.dpi)
    ax.barh(y=y, width=width, height=0.8, color=color)
    ax.set_title(title)
    ax.set_xlabel("Number of matches")
    ax.set_ylabel("Rule")
    save(fig, out_base)


def plot_annotation_progress(df: pd.DataFrame, label_col: str, out_base: Path, theme: PlotTheme, title: str) -> None:
    if label_col not in df.columns:
        return
    y = normalize_binary_label(df[label_col])
    annotated = y.notna()
    n_annot = int(annotated.sum())
    n_pending = int((~annotated).sum())
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=theme.dpi)

    axes[0].bar(["Annotated", "Pending"], [n_annot, n_pending],
                color=[theme.colors["hybrid"], theme.colors["neutral"]])
    axes[0].set_title("Progress")
    axes[0].set_ylabel("Number of documents")
    for i, v in enumerate([n_annot, n_pending]):
        axes[0].text(i, v, str(v), ha="center", va="bottom", fontsize=10)

    axes[1].bar(["Negative", "Positive"], [n_neg, n_pos],
                color=[theme.colors["neg"], theme.colors["pos"]])
    axes[1].set_title("Annotated label distribution")
    axes[1].set_ylabel("Number of documents")
    for i, v in enumerate([n_neg, n_pos]):
        axes[1].text(i, v, str(v), ha="center", va="bottom", fontsize=10)

    fig.suptitle(title, y=1.02)
    save(fig, out_base)


def plot_municipality_support(df: pd.DataFrame, label_col: str, muni_col: str, out_base: Path, theme: PlotTheme) -> None:
    if muni_col not in df.columns or label_col not in df.columns:
        return

    tmp = df.copy()
    tmp[label_col] = normalize_binary_label(tmp[label_col])
    tmp = tmp[tmp[label_col].notna()]

    support = tmp.groupby(muni_col)[label_col].count().sort_values(ascending=False).head(20)
    munis = support.index.tolist()

    pos = tmp[tmp[label_col] == 1].groupby(muni_col)[label_col].count().reindex(munis).fillna(0).to_numpy()
    neg = tmp[tmp[label_col] == 0].groupby(muni_col)[label_col].count().reindex(munis).fillna(0).to_numpy()

    y = np.arange(len(munis), dtype=float)

    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=theme.dpi)
    ax.barh(y=y, width=np.asarray(neg, dtype=float), height=0.8, color=theme.colors["neg"], label="Negative")
    ax.barh(y=y, width=np.asarray(pos, dtype=float), height=0.8, left=np.asarray(neg, dtype=float),
            color=theme.colors["pos"], label="Positive")

    ax.set_yticks(y)
    ax.set_yticklabels(munis)
    ax.set_xlabel("Number of gold-labeled documents")
    ax.set_ylabel("Municipality")
    ax.set_title("Gold labels per municipality (top 20)")
    ax.legend()

    save(fig, out_base)


# ============================================================
# Main
# ============================================================
@dataclass(frozen=True)
class Paths:
    base_dir: Path
    outdir: Path

    unlabeled_pool: Path
    weak_rules_labels: Path
    goldstandard_gold: Path
    annotation_master: Path
    annotation_round_extra30: Path


def resolve_paths(base_dir: Path, outdir: Path) -> Paths:
    return Paths(
        base_dir=base_dir,
        outdir=outdir,
        unlabeled_pool=base_dir / "unlabeled_pool.csv",
        weak_rules_labels=base_dir / "weak_rules_labels.csv",
        goldstandard_gold=base_dir / "goldstandard_gold.csv",
        annotation_master=base_dir / "annotation_master.csv",
        annotation_round_extra30=base_dir / "annotation_round_extra30.csv",
    )


def main(base_dir: str, outdir: str) -> None:
    theme = default_theme()
    apply_theme(theme)

    paths = resolve_paths(Path(base_dir), Path(outdir))
    ensure_outdir(paths.outdir)

    # Load
    df_pool = read_csv_auto(paths.unlabeled_pool)
    df_weak = read_csv_auto(paths.weak_rules_labels)
    df_gold = read_csv_auto(paths.goldstandard_gold)
    df_ann_master = read_csv_auto(paths.annotation_master)
    df_ann_extra = read_csv_auto(paths.annotation_round_extra30)

    # 01 Dataset sizes
    plot_dataset_sizes(
        items=[
            ("unlabeled_pool", len(df_pool)),
            ("weak_rules_labels", len(df_weak)),
            ("goldstandard_gold", len(df_gold)),
            ("annotation_master", len(df_ann_master)),
            ("annotation_extra30", len(df_ann_extra)),
        ],
        out_base=paths.outdir / "01_dataset_sizes",
        theme=theme,
    )

    # 02 Label balance
    gold_lbl = "gold_is_policy_measure" if "gold_is_policy_measure" in df_gold.columns else (
        "is_policy_measure" if "is_policy_measure" in df_gold.columns else None
    )
    if gold_lbl is not None:
        gold_y = normalize_binary_label(df_gold[gold_lbl])
        gold_pos = int((gold_y == 1).sum())
        gold_neg = int((gold_y == 0).sum())
        gold_na = int(gold_y.isna().sum())
    else:
        gold_pos = gold_neg = 0
        gold_na = len(df_gold)

    if "rule_label" in df_weak.columns:
        s = df_weak["rule_label"]
        weak_pos = int((s == 1).sum())
        weak_neg = int((s == 0).sum())
        weak_na = int(s.isna().sum())
    else:
        weak_pos = weak_neg = 0
        weak_na = len(df_weak)

    if "is_policy_measure" in df_ann_master.columns:
        ann_y = normalize_binary_label(df_ann_master["is_policy_measure"])
        ann_pos = int((ann_y == 1).sum())
        ann_neg = int((ann_y == 0).sum())
        ann_na = int(ann_y.isna().sum())
    else:
        ann_pos = ann_neg = 0
        ann_na = len(df_ann_master)

    plot_label_balance(
        rows=[
            ("Gold", gold_pos, gold_neg, gold_na),
            ("Weak rules", weak_pos, weak_neg, weak_na),
            ("Annotation master", ann_pos, ann_neg, ann_na),
        ],
        out_base=paths.outdir / "02_label_balance",
        theme=theme,
    )

    # 03 Text length distribution
    pool_text_col = get_text_col(df_pool)
    weak_text_col = get_text_col(df_weak)
    gold_text_col = get_text_col(df_gold)

    series_list = []
    labels = []
    if pool_text_col:
        series_list.append(df_pool[pool_text_col])
        labels.append("unlabeled_pool")
    if weak_text_col:
        series_list.append(df_weak[weak_text_col])
        labels.append("weak_rules_labels")
    if gold_text_col:
        series_list.append(df_gold[gold_text_col])
        labels.append("goldstandard_gold")

    if series_list:
        plot_text_length_distribution(
            series_list=series_list,
            labels=labels,
            out_base=paths.outdir / "03_text_length_distribution",
            theme=theme,
        )

    # 04 Weak rule coverage
    plot_weak_rule_coverage(df_weak, paths.outdir / "04_weak_rules_coverage", theme)

    # 05 Rule score distribution
    plot_rule_score_distribution(df_weak, paths.outdir / "05_rule_score_distribution", theme)

    # 06-07 Top rules
    freqs_all = extract_rule_frequencies(df_weak)
    plot_top_rules(
        freqs=freqs_all,
        title="Most frequent matched rules (overall)",
        color=theme.colors["rules"],
        out_base=paths.outdir / "06_top_rules_overall",
        theme=theme,
    )

    if "rule_label" in df_weak.columns:
        freqs_pos = extract_rule_frequencies(df_weak[df_weak["rule_label"] == 1])
        plot_top_rules(
            freqs=freqs_pos,
            title="Most frequent matched rules (positive-labeled documents only)",
            color=theme.colors["pos"],
            out_base=paths.outdir / "07_top_rules_positive_only",
            theme=theme,
        )

    # 08 Annotation progress (master)
    if "is_policy_measure" in df_ann_master.columns:
        plot_annotation_progress(
            df_ann_master,
            label_col="is_policy_measure",
            out_base=paths.outdir / "08_annotation_progress_master",
            theme=theme,
            title="Annotation progress (annotation_master.csv)",
        )

    # 09 Annotation progress (extra30)
    if "annotated_is_policy_measure" in df_ann_extra.columns:
        plot_annotation_progress(
            df_ann_extra,
            label_col="annotated_is_policy_measure",
            out_base=paths.outdir / "09_annotation_progress_extra30",
            theme=theme,
            title="Annotation progress (annotation_round_extra30.csv)",
        )
    elif "is_policy_measure" in df_ann_extra.columns:
        plot_annotation_progress(
            df_ann_extra,
            label_col="is_policy_measure",
            out_base=paths.outdir / "09_annotation_progress_extra30",
            theme=theme,
            title="Annotation progress (annotation_round_extra30.csv)",
        )

    # 10 Municipality support (gold)
    muni_col = get_muni_col(df_gold)
    if muni_col and gold_lbl:
        plot_municipality_support(
            df_gold,
            label_col=gold_lbl,
            muni_col=muni_col,
            out_base=paths.outdir / "10_municipality_support_gold",
            theme=theme,
        )

    print(f"Saved figures to: {paths.outdir.resolve()}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True, help="Path to data/de_by/goldstandard/")
    ap.add_argument("--outdir", type=str, default="results/figures/pipeline_full", help="Output directory for figures")
    args = ap.parse_args()

    main(base_dir=args.base_dir, outdir=args.outdir)
