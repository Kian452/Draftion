"""
Decides which win-probability model answers /recommend, in this order:

  1. WIN_MODEL_PATH set: exactly that file, nothing else (an explicit override,
     used by the smoke test).
  2. The newest trained model (model.resolve_model_path) if it was trained on real
     data, on at least MIN_TRAIN_ROWS battles, and does not lose to the raw-stats
     fallback in the cross-validation its training run recorded (see below).
  3. Otherwise the raw-stats fallback (baseline.py), built right now from whatever
     real battles are in the database - if there are at least MIN_RAW_BATTLES.
     The UI flags it as preliminary / low confidence. This beats a model trained
     on a few hundred battles, and beats the synthetic demo model as soon as any
     real data exists.
  4. Otherwise whatever trained model there is - the synthetic demo model that
     ships with the app (the UI flags it as a demo), or None (then /recommend
     answers 503).

"Not losing to the fallback" (2.) is measured, not assumed: on ~1.5k real battles the
simple per-brawler win rates scored a better forward-in-time log loss (0.685) than
the trained model with its thousands of pair/matchup features (0.691) - those
features only add noise until there is a lot more data. model.select_c records both
numbers on the same folds (metrics["cv_logloss"] / ["cv_raw_logloss"]); the trained
model takes over on its own as soon as it wins, with no threshold to keep in sync.

The fallback is rebuilt on every selection, so callers (api.reload_model) re-run
this after each crawl to let it follow the growing data.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from .baseline import RawStatsModel
from .config import Settings
from .extract import build_training_table
from .model import WinProbabilityModel, resolve_model_path

log = logging.getLogger("model_select")


@dataclass
class Selection:
    model: Optional[Any]
    # The trained-model file looked at (None if there was none). The API compares it
    # against the newest file on disk to notice a newer version - even while the
    # fallback is what's being served.
    trained_path: Optional[str]


def _raw_fallback(s: Settings) -> Optional[RawStatsModel]:
    if not os.path.exists(s.db_path):
        return None
    try:
        df = build_training_table(s.db_path, s.ranked_type, s.min_rank)
    except Exception:
        log.exception("Could not build the raw-stats fallback from %s.", s.db_path)
        return None
    if len(df) < s.min_raw_battles:
        return None
    return RawStatsModel.from_table(df)


def beats_raw_stats(model: WinProbabilityModel) -> bool:
    """False only if the training run measured the trained model as worse than the raw-stats
    fallback on the same cross-validation folds. No recorded comparison (a pinned C, an
    older pickle) means no evidence against it."""
    metrics = getattr(model, "metrics", None) or {}
    if "cv_logloss" not in metrics or "cv_raw_logloss" not in metrics:
        return True
    return metrics["cv_logloss"] <= metrics["cv_raw_logloss"]


def select_model(s: Settings) -> Selection:
    path = resolve_model_path(s.model_path, s.model_dir, s.legacy_model_path)
    trained = WinProbabilityModel.load(path) if path else None  # a missing explicit path raises FileNotFoundError
    if s.model_path:
        return Selection(trained, path)

    real = trained is not None and trained.data_source == "real"
    if real and trained.n_battles >= s.min_train_rows and beats_raw_stats(trained):
        return Selection(trained, path)

    raw = _raw_fallback(s)
    if raw is not None:
        if trained is None:
            why = "no trained model yet"
        elif not real:
            why = "the trained model is a synthetic demo"
        elif trained.n_battles < s.min_train_rows:
            why = f"the trained model has {trained.n_battles} battles, needs >= {s.min_train_rows}"
        else:
            m = trained.metrics
            why = (f"the trained model does not beat it yet (CV log loss {m['cv_logloss']:.4f} vs "
                   f"{m['cv_raw_logloss']:.4f} for the raw stats)")
        log.info("Serving the raw-stats fallback (%d real battles): %s.", raw.n_battles, why)
        return Selection(raw, path)
    return Selection(trained, path)


def confidence(model: Any, s: Settings) -> Optional[str]:
    """How much the UI should trust the model, one word: 'demo' (synthetic data),
    'low' (raw-stats fallback), 'early' (trained, but on fewer than EARLY_MODEL_BATTLES
    battles), else 'ok'. Measured: AUC 0.53-0.54 at ~2.8k battles, 0.58 at ~18k, and
    the trained model beat the raw stats by only 0.001 log loss there - so the
    warning stays until the ~40k that a brawler-level model is estimated to need."""
    if model is None:
        return None
    if getattr(model, "data_source", "real") != "real":
        return "demo"
    if getattr(model, "kind", "trained") == "raw-stats":
        return "low"
    return "early" if getattr(model, "n_battles", 0) < s.early_model_battles else "ok"
