"""
Empirical calibration of the training knobs on the data collected so far, so
`weight_step`, the regularization strength `C`, `min_rank` and the probability
calibration are chosen from measurements instead of left at their defaults.

    python -m src.tune                        # grid + fallback comparison
    python -m src.tune --folds 5 --eval-min-rank 12

How it is measured: expanding-window time-series cross-validation. The battles
are sorted by time and cut into `folds + 1` blocks; fold k trains on blocks
0..k and is scored on block k+1, so every score looks forward in time (a random
split would leak the future into the past, see model.py). Each fold is fitted
with model.fit_pipeline - the very code that trains the shipped model, including
its calibration slice - so the numbers describe what gets deployed. All settings
are scored on the SAME test battles (`--eval-min-rank` and up); otherwise a
config with a stricter `min_rank` would just be graded on an easier population.
Metrics are log loss / Brier / AUC on the plain (unweighted) outcome; the
weights only steer training. Always saying 50% scores a log loss of 0.6931 and a
Brier score of 0.25 - anything above that is worse than not modelling at all.

Also compares the trained model against the raw-statistics fallback (baseline.py)
for growing training sizes - what the `MIN_TRAIN_ROWS` threshold (below which the
app serves the fallback and says "preliminary") is judged against.

With a few thousand battles the differences between settings are small and
noisy; the table prints the standard error over folds so a "win" inside the
noise isn't mistaken for a real gain. Re-run it whenever the data has grown a
lot - the best values move with the amount of data.
"""
from __future__ import annotations

import argparse
import itertools
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .baseline import RawStatsModel
from .extract import build_training_table, rank_weight
from .features import encode_dataframe_rows
from .model import C_CANDIDATES, DEFAULT_C, evaluate, fit_pipeline, time_series_folds

log = logging.getLogger("tune")


def _scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    return {"logloss": float(log_loss(y, p, labels=[0, 1])), "brier": float(brier_score_loss(y, p)), "auc": auc}


def _summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("logloss", "brier", "auc"):
        vals = np.array([r[key] for r in rows])
        out[key] = float(np.nanmean(vals))
        out[key + "_se"] = float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan")
    return out


def grid(db_path: str, battle_type: str, folds: int, eval_min_rank: int, min_ranks: list[int],
         weight_steps: list[float], cs: list[float], calibrations: list[str]) -> pd.DataFrame:
    lowest = min(min_ranks + [eval_min_rank])
    base = build_training_table(db_path, battle_type, min_rank=lowest)
    log.info("%d battles at rank >= %d", len(base), lowest)
    results = []
    for min_rank, step, c, calibrate in itertools.product(min_ranks, weight_steps, cs, calibrations):
        fold_scores = []
        for train, test in time_series_folds(base, folds):
            train = train[train["rank"] >= min_rank].copy()
            test = test[test["rank"] >= eval_min_rank]
            if len(train) < 50 or len(test) < 30:
                continue
            train["weight"] = [rank_weight(r, min_rank, step) for r in train["rank"]]
            vec, est = fit_pipeline(train, "logreg", c, calibrate)
            fold_scores.append(evaluate(vec, est, test))
        if fold_scores:
            results.append({"min_rank": min_rank, "weight_step": step, "C": c, "calib": calibrate,
                            **_summarize(fold_scores)})
    return pd.DataFrame(results).sort_values("logloss")


def fallback_comparison(db_path: str, battle_type: str, min_rank: int, sizes: list[int], c: float,
                        test_fraction: float = 0.2) -> pd.DataFrame:
    """Raw-stats fallback vs the trained logreg on one fixed, later test block, for growing training sizes."""
    df = build_training_table(db_path, battle_type, min_rank).sort_values("battle_time").reset_index(drop=True)
    cut = int(len(df) * (1 - test_fraction))
    pool, test = df.iloc[:cut], df.iloc[cut:]
    y = test["winner"].to_numpy()
    situations = list(zip(test["team_a"], test["team_b"]))
    rows = []
    for n in sizes:
        if n > len(pool):
            break
        train = pool.iloc[len(pool) - n:]  # the n battles closest to the test block
        vec, est = fit_pipeline(train, "logreg", c, "sigmoid")
        p_model = est.predict_proba(vec.transform(encode_dataframe_rows(test)))[:, 1]
        row = {"train_battles": n, "test_battles": len(test), **{f"model_{k}": v for k, v in _scores(y, p_model).items()}}
        # the fallback has two knobs of its own; report the best-scoring pair so it isn't a strawman
        best = None
        for prior, scale in itertools.product((20.0, 50.0, 100.0, 200.0), (0.25, 0.5, 1.0)):
            raw = RawStatsModel.from_table(train, prior_games=prior, scale=scale)
            s = _scores(y, np.array(raw.predict_proba_many(None, situations)))
            if best is None or s["logloss"] < best[0]["logloss"]:
                best = (s, prior, scale)
        row.update({f"raw_{k}": v for k, v in best[0].items()}, raw_prior=best[1], raw_scale=best[2])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Calibrate weight_step / C / min_rank with time-series CV.")
    p.add_argument("--db", default="data/brawl.db")
    p.add_argument("--battle-type", default="soloRanked")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--eval-min-rank", type=int, default=10, help="Lobby rank of the battles every config is scored on.")
    p.add_argument("--min-ranks", type=int, nargs="+", default=[10, 12, 13])
    p.add_argument("--weight-steps", type=float, nargs="+", default=[0.0, 1.0])
    p.add_argument("--cs", type=float, nargs="+", default=[0.005, *C_CANDIDATES])
    p.add_argument("--calibrations", nargs="+", default=["none", "sigmoid"])
    p.add_argument("--sizes", type=int, nargs="+", default=[250, 500, 1000, 1500, 2000, 3000, 5000])
    p.add_argument("--top", type=int, default=15, help="Rows of the grid to print.")
    args = p.parse_args()

    pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:.4f}")
    print("\n== Grid (expanding-window CV; lower log loss is better; *_se = standard error over folds) ==")
    table = grid(args.db, args.battle_type, args.folds, args.eval_min_rank,
                 args.min_ranks, args.weight_steps, args.cs, args.calibrations)
    print(table.head(args.top).to_string(index=False))
    if not table.empty:
        best = table.iloc[0]
        print(f"\nbest: min_rank={int(best.min_rank)} weight_step={best.weight_step} C={best.C} "
              f"calib={best.calib} logloss={best.logloss:.4f} (+/-{best.logloss_se:.4f})")
        old = table[(table.min_rank == 10) & (table.weight_step == 1.0) & (table.C == 1.0) & (table.calib == "sigmoid")]
        if not old.empty:
            d = old.iloc[0]
            # NOT the pipeline of the first release: that one also cut a 15% slice off the training
            # data to Platt-calibrate on (0.702 measured then); calibration is skipped on this little
            # data now, so this is the sklearn-default C=1.0 fitted as the code trains today.
            print(f"sklearn default C=1.0 (min_rank 10, weight_step 1.0), fitted as trained today: "
                  f"logloss={d.logloss:.4f} (+/-{d.logloss_se:.4f})")

    print(f"\n== Trained model (C={DEFAULT_C}) vs raw-stats fallback, same later test block (min_rank={args.eval_min_rank}) ==")
    print(fallback_comparison(args.db, args.battle_type, args.eval_min_rank, args.sizes, DEFAULT_C).to_string(index=False))


if __name__ == "__main__":
    main()
