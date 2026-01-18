import argparse
from pathlib import Path

from bootstrap_core import (
    load_gold,
    split_gold,
    train_baseline,
    evaluate,
    load_weak_pool,
    weak_pool_stats,
    build_bootstrap_trainset,
    save_json,
    predict_table,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--weak", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_frac", type=float, default=0.30)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="gold_is_policy_measure")
    parser.add_argument("--min_pos_score", type=int, default=3)
    parser.add_argument("--max_neg_score", type=int, default=0)



    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    text_col = args.text_col
    label_col = args.label_col

    # --- load & split gold
    gold = load_gold(args.gold)
    gold_train, gold_test, split_meta = split_gold(
        gold, seed=args.seed, test_frac=args.test_frac, label_col=label_col
    )
    save_json(split_meta.__dict__, args.outdir / "split_meta.json")

    # --- baseline (train on gold_train, eval on gold_test)
    baseline_clf, baseline_vec = train_baseline(
        gold_train, text_col=text_col, label_col=label_col
    )
    baseline_metrics = evaluate(
        baseline_clf, baseline_vec, gold_test, text_col=text_col, label_col=label_col
    )
    save_json(baseline_metrics, args.outdir / "baseline_metrics.json")

    baseline_preds = predict_table(
        baseline_clf, baseline_vec, gold_test, text_col=text_col, label_col=label_col
    )
    baseline_preds.to_csv(args.outdir / "baseline_predictions.csv", index=False)

    # --- weak pool
    weak_pool = load_weak_pool(args.weak)

    # stats before leakage filtering
    save_json(weak_pool_stats(weak_pool), args.outdir / "weak_pool_stats_before.json")

    # leakage filter: remove weak examples that are in gold_test by candidate_id
    test_ids = set(gold_test["candidate_id"].astype(str))
    weak_pool["candidate_id"] = weak_pool["candidate_id"].astype(str)
    weak_pool = weak_pool[~weak_pool["candidate_id"].isin(test_ids)].reset_index(drop=True)

    # stats after leakage filtering
    save_json(weak_pool_stats(weak_pool), args.outdir / "weak_pool_stats_after.json")

    # --- bootstrap trainset (gold_train + weak labels), then eval on gold_test
    bootstrap_train = build_bootstrap_trainset(
        gold_train,
        weak_pool,
        text_col=text_col,
        gold_label_col=label_col,
        min_pos_score=args.min_pos_score,
        max_neg_score=args.max_neg_score,
    )


    bootstrap_clf, bootstrap_vec = train_baseline(
        bootstrap_train, text_col=text_col, label_col="label"
    )

    bootstrap_metrics = evaluate(
        bootstrap_clf, bootstrap_vec, gold_test, text_col=text_col, label_col=label_col
    )
    save_json(bootstrap_metrics, args.outdir / "bootstrap_metrics.json")

    bootstrap_preds = predict_table(
        bootstrap_clf, bootstrap_vec, gold_test, text_col=text_col, label_col=label_col
    )
    bootstrap_preds.to_csv(args.outdir / "bootstrap_predictions.csv", index=False)

    # --- run config (avoid Path objects)
    run_config = {
        "gold": str(args.gold),
        "weak": str(args.weak),
        "seed": args.seed,
        "test_frac": args.test_frac,
        "outdir": str(args.outdir),
        "text_col": text_col,
        "label_col": label_col,
    }
    save_json(run_config, args.outdir / "run_config.json")


if __name__ == "__main__":
    main()
