"""
Win-probability model (Abschnitt 8): P(Team A wins | map, Team A, Team B),
also usable with partial (in-draft) teams.

Two backends, both trained on top of the same sparse feature encoding
(features.encode_situation) so they're directly comparable:

    logreg  - L2-regularized logistic regression. The spec calls out that a
              naive "winrate with brawler X" count confounds real synergy
              with X's general strength; L2 regularization is what
              disentangles that here (Abschnitt 6/8), not raw counting.
    gbm     - LightGBM gradient boosting. Tends to find interactions via
              splits automatically; kept as the second approach the spec
              asks to evaluate against logreg.

Training:
    python -m src.model train --backend logreg --min-rank 12 --battle-type soloRanked

Every training run writes a NEW versioned file (data/models/win_model_<UTC time>.joblib,
newest few kept). The API always serves the newest version and notices a newer one
without a restart, so retraining - by hand, by a cron job or by the in-service worker
(serve.py) - never needs the server touched. See save_versioned() / resolve_model_path().

- sample_weight comes straight from extract.rank_weight() (Abschnitt 5).
- The train/test split is by TIME, not random (Abschnitt 8): the meta shifts
  with balance patches, so a random split would leak future information into
  the past and overstate performance.
- Reports calibration (predicted-probability bins vs actual win rate) plus
  Brier score and log loss, not just accuracy - Abschnitt 8 is explicit that
  a well-calibrated probability matters more than raw hit rate for a
  recommendation engine.
- Applies a post-hoc calibration step (Platt scaling by default, isotonic as an
  alternative) fit on a held-out calibration slice carved out of the training
  period - Abschnitt 8's explicit fallback for when the raw model output isn't
  well-calibrated on its own. Skipped automatically when the slice is too small
  (see _MIN_CALIBRATION_SAMPLES: measured to make the model worse below ~1000
  battles), which is the normal case while the crawl is young - then the whole
  training period is used to fit the model.
- The regularization strength C is chosen per training run by time-series CV
  (select_c) and recorded with the held-out test metrics, the CV log loss and the
  raw-stats fallback's CV log loss on the same folds in `model.metrics`.
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .extract import build_training_table
from .features import encode_dataframe_rows, encode_situation

log = logging.getLogger("model")

DEFAULT_MODEL_PATH = "data/win_model.joblib"  # the old single-file location; still loaded if no versions exist
DEFAULT_MODEL_DIR = "data/models"
MODEL_FILE_PREFIX = "win_model_"

# Regularization strength of the logreg (sklearn's C: smaller = stronger), used only
# when there are too few battles for select_c's cross-validation. Measured with
# `python -m src.tune` on ~2.8k real battles (forward-in-time log loss; always saying
# 50% scores 0.6931): C=1.0, the sklearn default, gives 0.95 uncalibrated (0.70 after
# a Platt fit) - it memorizes noise - while a strong penalty (0.01-0.02) gives 0.691:
# the model stays near 50% and only moves where the data insists.
DEFAULT_C = 0.02


@dataclass
class WinProbabilityModel:
    backend: str
    vectorizer: DictVectorizer
    estimator: Any
    brawlers: list[str]
    # "synthetic" = trained on src/synthetic.py's fake data, so the UI can warn
    # that its recommendations are placeholders and not based on real matches.
    data_source: str = "real"
    # Battles the model was trained on (0 = unknown, e.g. an older pickle) - lets the
    # UI say that an early model trained on a few thousand battles is still noisy.
    n_battles: int = 0
    # UTC ISO time of training and the held-out (time-based) test metrics from that
    # run: {"auc", "brier", "logloss", "n_test", "c"} plus, when C was chosen by CV,
    # "cv_logloss" and "cv_raw_logloss" (the fallback on the same folds). Empty for
    # older pickles.
    trained_at: str = ""
    metrics: dict = field(default_factory=dict)

    kind = "trained"

    def known_brawlers(self) -> set[str]:
        return {n.split(":", 1)[1] for n in self.vectorizer.feature_names_ if n.startswith("brawler_a:")}

    def predict_proba(
        self,
        map_name: Optional[str],
        team_a: Sequence[Optional[str]],
        team_b: Sequence[Optional[str]],
    ) -> float:
        """P(Team A wins). Works with partial teams (fewer than 3 slots filled)."""
        return self.predict_proba_many(map_name, [(team_a, team_b)])[0]

    def predict_proba_many(
        self,
        map_name: Optional[str],
        situations: Sequence[tuple[Sequence[Optional[str]], Sequence[Optional[str]]]],
    ) -> list[float]:
        """Batched predict_proba: one vectorizer + estimator call for all
        situations, several times cheaper than calling predict_proba in a loop
        (the draft search evaluates hundreds of candidates per step)."""
        X = self.vectorizer.transform([encode_situation(map_name, a, b) for a, b in situations])
        return self.estimator.predict_proba(X)[:, 1].tolist()

    def save(self, path: str = DEFAULT_MODEL_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = DEFAULT_MODEL_PATH) -> "WinProbabilityModel":
        return joblib.load(path)


def list_versions(model_dir: str = DEFAULT_MODEL_DIR) -> list[Path]:
    """Saved model versions, oldest first (the UTC timestamp in the file name sorts chronologically)."""
    return sorted(Path(model_dir).glob(f"{MODEL_FILE_PREFIX}*.joblib"))


def latest_version(model_dir: str = DEFAULT_MODEL_DIR) -> Optional[Path]:
    versions = list_versions(model_dir)
    return versions[-1] if versions else None


def save_versioned(model: WinProbabilityModel, model_dir: str = DEFAULT_MODEL_DIR, keep: int = 5) -> str:
    """Write `model` as a new version file and drop all but the newest `keep`.
    Written to a temp name and renamed, so a server that is scanning the directory
    never sees (and tries to load) a half-written file."""
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(model_dir) / f"{MODEL_FILE_PREFIX}{stamp}.joblib"
    tmp = path.with_suffix(".tmp")
    joblib.dump(model, tmp)
    os.replace(tmp, path)
    for old in list_versions(model_dir)[:-max(keep, 1)]:
        try:
            old.unlink()
        except OSError:
            log.warning("Could not delete old model version %s", old)
    return str(path)


def resolve_model_path(
    explicit: Optional[str] = None,
    model_dir: str = DEFAULT_MODEL_DIR,
    legacy_path: str = DEFAULT_MODEL_PATH,
) -> Optional[str]:
    """Which trained-model file to serve: an explicitly configured path (WIN_MODEL_PATH)
    wins; otherwise the newest version in `model_dir`; otherwise the old single file.
    None if there is no model at all."""
    if explicit:
        return explicit
    latest = latest_version(model_dir)
    if latest is not None:
        return str(latest)
    return legacy_path if Path(legacy_path).exists() else None


def _build_estimator(backend: str, c: float = DEFAULT_C):
    if backend == "logreg":
        # Sparse indicator features on the same 0/1 scale - no need to
        # standardize. L2 is the confound-control mechanism called out in
        # Abschnitt 6/8; its strength `c` was measured, see DEFAULT_C.
        return LogisticRegression(max_iter=2000, C=c)
    if backend == "gbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as e:
            raise RuntimeError(
                "backend='gbm' needs lightgbm installed (pip install lightgbm)."
            ) from e
        return LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05)
    raise ValueError(f"Unknown backend: {backend!r} (expected 'logreg' or 'gbm')")


def time_based_split(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort by battle_time and take the trailing `test_fraction` as the held-out
    set, so evaluation always looks forward in time relative to training -
    never randomly shuffled (see module docstring)."""
    ordered = df.sort_values("battle_time").reset_index(drop=True)
    split_at = int(len(ordered) * (1 - test_fraction))
    return ordered.iloc[:split_at], ordered.iloc[split_at:]


def _with_mirrored(df: pd.DataFrame) -> pd.DataFrame:
    """The same battles seen from the other side: teams swapped, winner flipped."""
    mirrored = df.copy()
    mirrored["team_a"], mirrored["team_b"] = df["team_b"], df["team_a"]
    mirrored["winner"] = 1 - df["winner"]
    return pd.concat([df, mirrored], ignore_index=True)


def calibration_report(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_pred, bins) - 1, 0, n_bins - 1)
    rows = []
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": f"[{bins[i]:.1f}, {bins[i+1]:.1f})",
            "n": int(mask.sum()),
            "predicted_mean": float(y_pred[mask].mean()),
            "actual_winrate": float(y_true[mask].mean()),
        })
    return pd.DataFrame(rows)



# Isotonic regression is non-parametric (a step function fit to the sorted
# calibration predictions) and needs substantially more data than Platt
# scaling (a smooth 2-parameter logistic fit) to avoid overfitting. This was
# not a theoretical concern: on this project's own synthetic smoke-test
# calibration slice (~480 rows), isotonic calibration collapsed a clear,
# correctly-learned signal - P(win | BULL vs CROW) dropped from a
# pre-calibration 0.983 vs 0.760 (BULL correctly favored as the counter
# pick) down to a post-calibration 0.680 vs 0.679, i.e. statistically
# indistinguishable. That's exactly the kind of fine-grained, low-sample-count
# distinction the draft search (Abschnitt 9) depends on to tell candidates
# apart, so an isotonic calibrator that smooths it away is actively harmful
# here even though it may well improve *aggregate* calibration metrics.
# Hence: much stricter minimum sample count for isotonic than for sigmoid,
# and sigmoid - not isotonic - is the default.
# Sigmoid: measured on ~2.8k real battles with `python -m src.tune` (time-series CV).
# With a strongly regularized model (the default, see DEFAULT_C) a Platt fit on the
# ~330 battles of a 15% calibration slice made forward-in-time log loss WORSE by
# 0.002-0.005 (it stretches the model's small, noisy signal); it only paid off for
# weakly regularized models. So it stays off until the slice is large enough to
# estimate its slope reliably. Counts are real battles per class (the slice is
# mirrored, so each battle contributes one positive and one negative).
_MIN_CALIBRATION_SAMPLES = {"sigmoid": 1000, "isotonic": 1000}


def _fit_calibrated(estimator, X_calib, y_calib, w_calib, method: str):
    """
    Wraps a fitted estimator with post-hoc probability calibration
    (Abschnitt 8), fit on a held-out calibration slice - never on the same
    data the base estimator was trained on, or the calibration would just
    memorize training noise instead of correcting genuine miscalibration.

    Returns the (possibly wrapped) estimator, or the original one unchanged
    if the calibration slice is too small/single-class for the chosen
    method to fit safely (see _MIN_CALIBRATION_SAMPLES above).
    """
    if method == "none":
        return estimator
    n_pos = int((y_calib == 1).sum())
    n_neg = int((y_calib == 0).sum())
    min_needed = _MIN_CALIBRATION_SAMPLES[method]
    if min(n_pos, n_neg) < min_needed:
        # Expected, not alarming, while there is little data (see _MIN_CALIBRATION_SAMPLES),
        # and it is hit once per fit during select_c's cross-validation - hence debug;
        # train_model reports the outcome once.
        log.debug("Calibration slice too small for method=%r (pos=%d, neg=%d, need >=%d each) - skipped.",
                  method, n_pos, n_neg, min_needed)
        return estimator
    # FrozenEstimator marks `estimator` as already-fitted, so
    # CalibratedClassifierCV only fits the calibration mapping itself on
    # (X_calib, y_calib) instead of refitting/cross-validating the base
    # estimator (the old cv="prefit" API this replaces in modern sklearn).
    from sklearn.frozen import FrozenEstimator

    calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method=method)
    calibrated.fit(X_calib, y_calib, sample_weight=w_calib)
    return calibrated


def _unit_mean(weights: np.ndarray) -> np.ndarray:
    """Rescale sample weights to mean 1. sklearn's C is fixed against the *sum* of the
    weighted losses, so weights averaging 3 silently act like a 3x larger C (weaker
    regularization). Measured: that alone made rank weighting look harmful. Normalized,
    weight_step only changes which battles count more, not how hard the model is
    regularized."""
    mean = float(weights.mean()) if len(weights) else 0.0
    return weights / mean if mean > 0 else weights


def fit_pipeline(
    train_df: pd.DataFrame,
    backend: str = "logreg",
    c: float = DEFAULT_C,
    calibrate: str = "sigmoid",
    calibration_fraction: float = 0.15,
) -> tuple[DictVectorizer, Any]:
    """The whole fit as production does it: carve a calibration slice off the END of
    `train_df` (time-ordered, still before any test period), fit the estimator on the
    rest, calibrate on the slice. Shared by train_model and the tuner (tune.py) so the
    tuner measures exactly what gets shipped."""
    # A time-ordered slice off the end of the training period to calibrate on -
    # calibrating on the rows the base estimator fit would just rediscover
    # training noise instead of correcting real miscalibration. If that slice would
    # be too small to calibrate with anyway (the normal case on little data, see
    # _MIN_CALIBRATION_SAMPLES), don't cut it off: a slice that is never used just
    # throws 15% of the training data away (measured: ~0.004 log loss at 1.5k battles).
    if calibrate == "none" or int(len(train_df) * calibration_fraction) < _MIN_CALIBRATION_SAMPLES[calibrate]:
        fit_df, calib_df = train_df, train_df.iloc[0:0]
    else:
        fit_df, calib_df = time_based_split(train_df, calibration_fraction)
    # Which team the API happens to list first carries no information, so the
    # model must not have a side preference. Train on every battle from both
    # sides (teams swapped, winner flipped): the fitted model is then symmetric
    # (P(A beats B) + P(B beats A) = 1), which the results screen and the draft
    # search rely on.
    fit_df, calib_df = _with_mirrored(fit_df), _with_mirrored(calib_df)

    vectorizer = DictVectorizer(sparse=True)
    X_fit = vectorizer.fit_transform(encode_dataframe_rows(fit_df))
    estimator = _build_estimator(backend, c)
    estimator.fit(X_fit, fit_df["winner"].to_numpy(), sample_weight=_unit_mean(fit_df["weight"].to_numpy()))

    if len(calib_df) > 0:
        X_calib = vectorizer.transform(encode_dataframe_rows(calib_df))
        estimator = _fit_calibrated(
            estimator, X_calib, calib_df["winner"].to_numpy(), _unit_mean(calib_df["weight"].to_numpy()), calibrate,
        )
    return vectorizer, estimator


def time_series_folds(df: pd.DataFrame, folds: int):
    """Expanding-window CV splits: sort by time, cut into `folds + 1` blocks; fold k
    trains on blocks 0..k and is scored on block k+1, so every score looks forward in
    time. Yields (train_df, test_df)."""
    ordered = df.sort_values("battle_time").reset_index(drop=True)
    edges = np.linspace(0, len(ordered), folds + 2, dtype=int)
    for k in range(folds):
        yield ordered.iloc[: edges[k + 1]], ordered.iloc[edges[k + 1] : edges[k + 2]]


# Candidate regularization strengths for select_c, weakest data-hunger first.
C_CANDIDATES = (0.01, 0.03, 0.1, 0.3, 1.0)


def select_c(
    df: pd.DataFrame,
    backend: str = "logreg",
    candidates: Sequence[float] = C_CANDIDATES,
    folds: int = 3,
    max_rows: int = 40_000,
    calibrate: str = "sigmoid",
    info: Optional[dict] = None,
) -> float:
    """Pick the L2 strength by time-series CV on `df`'s log loss. The best value
    depends on how much data there is: ~2.8k real battles wanted C=0.01 (larger values
    overfit noise, see DEFAULT_C), and it rises as the crawl grows - a constant would be
    right today and wrong in a month, so every training run re-measures it. Only the
    newest `max_rows` battles are used, which bounds the cost on a big database.
    Below 300 battles the CV is meaningless and DEFAULT_C is returned.

    If `info` is given it is filled with the winner's CV log loss (`cv_logloss`) and,
    measured on the very same folds, that of the raw-stats fallback (`cv_raw_logloss`,
    baseline.py). model_select uses the pair to never serve a trained model that is
    measurably worse than the fallback - so "there is still too little data" is
    decided by a measurement, not by a fixed number of battles."""
    if backend != "logreg" or len(df) < 300:
        return DEFAULT_C
    from .baseline import RawStatsModel

    recent = df.sort_values("battle_time").tail(max_rows)
    fold_pairs = list(time_series_folds(recent, folds))
    best_c, best_loss = DEFAULT_C, float("inf")
    for c in candidates:
        losses = []
        for train, test in fold_pairs:
            vec, est = fit_pipeline(train, backend, c, calibrate)
            losses.append(evaluate(vec, est, test)["logloss"])
        loss = float(np.mean(losses))
        log.info("  C=%-5g CV log loss %.4f", c, loss)
        if loss < best_loss:
            best_c, best_loss = c, loss
    if info is not None:
        raw_losses = []
        for train, test in fold_pairs:
            raw = RawStatsModel.from_table(train)
            p = np.clip(raw.predict_proba_many(None, list(zip(test["team_a"], test["team_b"]))), 1e-6, 1 - 1e-6)
            raw_losses.append(log_loss(test["winner"].to_numpy(), p, labels=[0, 1]))
        info.update(cv_logloss=best_loss, cv_raw_logloss=float(np.mean(raw_losses)))
        log.info("  raw-stats fallback on the same folds: CV log loss %.4f", info["cv_raw_logloss"])
    return best_c


def evaluate(vectorizer: DictVectorizer, estimator: Any, test_df: pd.DataFrame) -> dict[str, float]:
    """Held-out metrics: AUC, Brier, log loss (Abschnitt 8: calibration, not just hit rate)."""
    y_true = test_df["winner"].to_numpy()
    y_pred = estimator.predict_proba(vectorizer.transform(encode_dataframe_rows(test_df)))[:, 1]
    try:
        auc = float(roc_auc_score(y_true, y_pred))
    except ValueError:  # only one class in the test set
        auc = float("nan")
    return {
        "auc": auc,
        "brier": float(brier_score_loss(y_true, y_pred)),
        "logloss": float(log_loss(y_true, y_pred, labels=[0, 1])),
        "n_test": int(len(test_df)),
    }


def train_model(
    db_path: str,
    battle_type: str,
    min_rank: int,
    backend: str = "logreg",
    weight_step: float = 1.0,
    test_fraction: float = 0.2,
    calibrate: str = "sigmoid",
    calibration_fraction: float = 0.15,
    brawlers: Optional[list[str]] = None,
    data_source: str = "real",
    c: Optional[float] = None,
) -> tuple[WinProbabilityModel, pd.DataFrame]:
    """`c=None` (default) picks the regularization strength by CV on the training
    period (select_c); pass a number to pin it."""
    df = build_training_table(db_path, battle_type, min_rank, weight_step)
    if df.empty:
        raise RuntimeError(
            "No training rows found. Crawl more data first, or check "
            "--battle-type / --min-rank against `python -m src.extract types/ranks`."
        )
    if len(df) < 50:
        log.warning(
            "Only %d training rows - results will be noisy. This is fine for a "
            "pipeline smoke test, not for real recommendations yet.", len(df),
        )

    train_full_df, test_df = time_based_split(df, test_fraction)
    cv_info: dict = {}
    if c is None:
        log.info("Choosing the regularization strength C by time-series CV on the training period:")
        c = select_c(train_full_df, backend, calibrate=calibrate, info=cv_info)
        log.info("  -> C=%g", c)
    # The held-out test set stays as-is (not mirrored, not calibrated on, not used to
    # choose C) so the reported metrics are honest.
    vectorizer, estimator = fit_pipeline(train_full_df, backend, c, calibrate, calibration_fraction)
    if calibrate != "none":
        applied = isinstance(estimator, CalibratedClassifierCV)
        log.info("Probability calibration (%s): %s", calibrate,
                 "applied" if applied else
                 f"skipped - the calibration slice has fewer than {_MIN_CALIBRATION_SAMPLES[calibrate]} battles "
                 "(it measurably hurts below that, see _MIN_CALIBRATION_SAMPLES)")

    metrics: dict = {}
    if len(test_df) > 0:
        metrics = evaluate(vectorizer, estimator, test_df)
        metrics["c"] = float(c)
        metrics.update(cv_info)
        log.info("Held-out (time-based) evaluation, n=%d:", metrics["n_test"])
        log.info("  AUC        = %.4f", metrics["auc"])
        log.info("  Brier      = %.4f (lower is better)", metrics["brier"])
        log.info("  LogLoss    = %.4f (always saying 50%% gives 0.6931)", metrics["logloss"])
        y_pred = estimator.predict_proba(vectorizer.transform(encode_dataframe_rows(test_df)))[:, 1]
        cal = calibration_report(test_df["winner"].to_numpy(), y_pred)
        if not cal.empty:
            log.info("Calibration (predicted vs actual win rate per bin):\n%s", cal.to_string(index=False))
    else:
        log.warning("Test set is empty - skipping evaluation. Add more data or lower --test-fraction.")

    model = WinProbabilityModel(
        backend=backend, vectorizer=vectorizer, estimator=estimator,
        brawlers=brawlers or [], data_source=data_source, n_battles=len(df),
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), metrics=metrics,
    )
    return model, df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the win-probability model (Abschnitt 8).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument("--db", default="data/brawl.db")
    train_p.add_argument("--battle-type", default="soloRanked",
                          help="Confirm via `python -m src.extract types`.")
    train_p.add_argument("--min-rank", type=int, required=True,
                          help="Confirm via `python -m src.extract ranks` (Legendary III cutoff).")
    train_p.add_argument("--backend", choices=["logreg", "gbm"], default="logreg")
    train_p.add_argument("--weight-step", type=float, default=1.0)
    train_p.add_argument("--test-fraction", type=float, default=0.2)
    train_p.add_argument("--calibrate", choices=["isotonic", "sigmoid", "none"], default="sigmoid",
                          help="Post-hoc probability calibration (Abschnitt 8). 'sigmoid' = Platt scaling "
                               "(default - robust on small data); 'isotonic' needs a lot more calibration "
                               "data or it can collapse fine-grained signal, see model.py docstring.")
    train_p.add_argument("--calibration-fraction", type=float, default=0.15,
                          help="Share of the (pre-test) training period held out to fit the calibrator.")
    train_p.add_argument("--c", type=float, default=None,
                          help="Inverse L2 strength of the logreg (smaller = stronger). Default: chosen by "
                               "time-series CV on the current data (see select_c / `python -m src.tune`).")
    train_p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                          help="Directory for versioned model files (win_model_<UTC time>.joblib); the API serves the newest.")
    train_p.add_argument("--keep", type=int, default=5, help="How many model versions to keep.")
    train_p.add_argument("--out", default=None,
                          help="Write to exactly this file instead of a new version in --model-dir.")
    train_p.add_argument("--brawlers", default="data/brawlers.json",
                          help="Path from `python -m src.brawlers fetch`; used for the draft engine's candidate pool.")

    args = parser.parse_args()
    if args.cmd == "train":
        from .brawlers import load_brawlers
        brawlers = load_brawlers(args.brawlers)
        model, _ = train_model(
            args.db, args.battle_type, args.min_rank,
            backend=args.backend, weight_step=args.weight_step,
            test_fraction=args.test_fraction, calibrate=args.calibrate,
            calibration_fraction=args.calibration_fraction, brawlers=brawlers, c=args.c,
        )
        if args.out:
            model.save(args.out)
            log.info("Saved model to %s", args.out)
        else:
            log.info("Saved model version %s", save_versioned(model, args.model_dir, args.keep))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Run main() from the *imported* module, not from this `__main__` copy: classes
    # defined in `__main__` get pickled as `__main__.WinProbabilityModel`, which no
    # other process (the API server) can load - so a model trained with
    # `python -m src.model train` would be unusable.
    from src.model import main as _main

    _main()
