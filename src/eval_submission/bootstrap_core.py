from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Any, Dict

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression



# =========================
# Data containers
# =========================

@dataclass
class SplitMeta:
    seed: int
    test_frac: float
    n_train: int
    n_test: int
    pos_train: int
    pos_test: int


# =========================
# Utilities
# =========================

def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# =========================
# Core steps
# =========================

def load_gold(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "candidate_id" in df.columns, "gold CSV must contain 'candidate_id'"
    assert "gold_is_policy_measure" in df.columns, "gold CSV must contain 'gold_is_policy_measure'"
    return df



def split_gold(
    gold: pd.DataFrame,
    seed: int,
    test_frac: float,
    label_col: str = "gold_is_policy_measure",
) -> Tuple[pd.DataFrame, pd.DataFrame, SplitMeta]:

    train, test = train_test_split(
        gold,
        test_size=test_frac,
        random_state=seed,
        stratify=gold[label_col],
    )

    meta = SplitMeta(
        seed=seed,
        test_frac=test_frac,
        n_train=int(len(train)),
        n_test=int(len(test)),
        pos_train=int(train[label_col].sum()),
        pos_test=int(test[label_col].sum()),
    )
    return train.reset_index(drop=True), test.reset_index(drop=True), meta



def train_baseline(train_df: pd.DataFrame, text_col: str = "text", label_col: str = "gold_is_policy_measure"):
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1, 2))
    X = vectorizer.fit_transform(train_df[text_col].astype(str))
    y = train_df[label_col].to_numpy(dtype=int)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=0,
    )
    clf.fit(X, y)
    return clf, vectorizer




def evaluate(clf, vectorizer, test_df: pd.DataFrame, text_col: str = "text", label_col: str = "gold_is_policy_measure") -> dict:
    X = vectorizer.transform(test_df[text_col].astype(str))
    y_true = test_df[label_col].to_numpy(dtype=int)
    y_pred = clf.predict(X)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    return {
        "n": int(len(y_true)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((tp + tn) / len(y_true)) if len(y_true) > 0 else 0.0,
    }



def load_weak_pool(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "rule_label" in df.columns, "weak pool must contain 'rule_label'"
    return df


def weak_pool_stats(df: pd.DataFrame) -> Dict[str, Any]:
    n_total = int(len(df))
    n_pos = int((df["rule_label"] == 1).sum())
    n_neg = int((df["rule_label"] == 0).sum())
    n_abstain = int(df["rule_label"].isna().sum())
    coverage = (n_pos + n_neg) / n_total if n_total > 0 else 0.0

    return {
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_abstain": n_abstain,
        "coverage": float(coverage),
    }


def build_bootstrap_trainset(
    gold_train: pd.DataFrame,
    weak_pool: pd.DataFrame,
    text_col: str = "text",
    gold_label_col: str = "gold_is_policy_measure",
    min_pos_score: int = 3,
    max_neg_score: int = 0,
) -> pd.DataFrame:
    gold_part = gold_train.copy()
    gold_part["label"] = gold_part[gold_label_col].to_numpy(dtype=int)

    # Start from labeled only
    wp = weak_pool[weak_pool["rule_label"].isin([0, 1])].copy()

    # Conservative score filtering (if present)
    if "rule_score" in wp.columns:
        pos_mask = (wp["rule_label"] == 1) & (wp["rule_score"] >= min_pos_score)
        neg_mask = (wp["rule_label"] == 0) & (wp["rule_score"] <= max_neg_score)
        wp = wp[pos_mask | neg_mask].copy()

    wp["label"] = wp["rule_label"].to_numpy(dtype=int)

    if text_col != "text":
        wp[text_col] = wp["text"].astype(str)

    out_cols = ["candidate_id", text_col, "label"]
    return pd.concat([gold_part[out_cols], wp[out_cols]], ignore_index=True)


def predict_table(clf, vectorizer, df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    X = vectorizer.transform(df[text_col].astype(str))
    pred = clf.predict(X).astype(int)

    out = pd.DataFrame({
        "candidate_id": df["candidate_id"].astype(str),
        "gold_label": df[label_col].to_numpy(dtype=int),
        "pred_label": pred,
    })

    if hasattr(clf, "predict_proba"):
        out["score"] = clf.predict_proba(X)[:, 1]

    return out

