"""
All environment-variable configuration in one place, shared by the API (api.py),
the hosted runner (serve.py) and the model selection (model_select.py). Every
variable is optional; the defaults are what a local run and the first Railway
deployment want.

Data / files
  DB_PATH, BRAWLERS_PATH, MAPS_PATH   data/brawl.db, data/brawlers.json, data/maps.json
  WIN_MODEL_PATH       serve exactly this model file (no version lookup, no fallback)
  WIN_MODEL_DIR        directory of versioned models                       (data/models)
  LEGACY_MODEL_PATH    single-file model used when there are no versions   (data/win_model.joblib;
                       this is where the shipped synthetic demo model lives)
  KEEP_MODELS          versions kept after each training                   (5)

Ranked data
  RANKED_TYPE          battle type of Ranked drafts (BATTLE_TYPE also read) (soloRanked)
  MIN_RANK             lowest lobby rank to train on and to re-poll         (12)

Model quality (see `python -m src.tune` for how these were measured)
  MIN_TRAIN_ROWS       usable battles needed before a trained model is built or served;
                       below it the raw-stats fallback answers instead      (1000)
  MIN_RAW_BATTLES      battles needed for the raw-stats fallback            (100)
  EARLY_MODEL_BATTLES  a trained model below this is flagged "early"        (40000)
  DRAFT_TIME_BUDGET    seconds one /recommend search may take              (1.0)
  MODEL_CHECK_SECONDS  how often the API looks for a newer model version    (30)

Maps
  MAP_WINDOW_DAYS      a map is in the Ranked pool if played within this many days
                       of the newest crawled battle                          (7)
  MAP_MIN_BATTLES      ... at least this often (filters stray battles)      (3)

Hosted runner (serve.py)
  BS_API_TOKEN, BS_API_BASE_URL   the key / proxy base URL (see api_client.py)
  CRAWL_INTERVAL_HOURS, CRAWL_MAX_PLAYERS                                  (2, 1500)
  RETRAIN_INTERVAL_HOURS    how often to consider retraining               (24)
  RETRAIN_MIN_NEW_BATTLES   ... and only if this many new usable battles came in (100)
  TRAIN_BACKEND        logreg | gbm                                        (logreg)
  DRAFTION_WORKER      0 turns the background job off
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    db_path: str
    brawlers_path: str
    maps_path: str
    model_path: str | None
    model_dir: str
    legacy_model_path: str
    keep_models: int
    ranked_type: str
    min_rank: int
    min_train_rows: int
    min_raw_battles: int
    early_model_battles: int
    draft_time_budget: float
    model_check_seconds: float
    map_window_days: int
    map_min_battles: int
    crawl_interval_hours: float
    crawl_max_players: int
    retrain_interval_hours: float
    retrain_min_new_battles: int
    backend: str

    @staticmethod
    def from_env() -> "Settings":
        env = os.environ.get
        return Settings(
            db_path=env("DB_PATH", "data/brawl.db"),
            brawlers_path=env("BRAWLERS_PATH", "data/brawlers.json"),
            maps_path=env("MAPS_PATH", "data/maps.json"),
            model_path=env("WIN_MODEL_PATH") or None,
            model_dir=env("WIN_MODEL_DIR", "data/models"),
            legacy_model_path=env("LEGACY_MODEL_PATH", "data/win_model.joblib"),
            keep_models=int(env("KEEP_MODELS", "5")),
            ranked_type=env("RANKED_TYPE") or env("BATTLE_TYPE") or "soloRanked",
            min_rank=int(env("MIN_RANK", "12")),
            min_train_rows=int(env("MIN_TRAIN_ROWS", "1000")),
            min_raw_battles=int(env("MIN_RAW_BATTLES", "100")),
            early_model_battles=int(env("EARLY_MODEL_BATTLES", "40000")),
            draft_time_budget=float(env("DRAFT_TIME_BUDGET", "1.0")),
            model_check_seconds=float(env("MODEL_CHECK_SECONDS", "30")),
            map_window_days=int(env("MAP_WINDOW_DAYS", "7")),
            map_min_battles=int(env("MAP_MIN_BATTLES", "3")),
            crawl_interval_hours=float(env("CRAWL_INTERVAL_HOURS", "2")),
            crawl_max_players=int(env("CRAWL_MAX_PLAYERS", "1500")),
            retrain_interval_hours=float(env("RETRAIN_INTERVAL_HOURS", "24")),
            retrain_min_new_battles=int(env("RETRAIN_MIN_NEW_BATTLES", "100")),
            backend=env("TRAIN_BACKEND", "logreg"),
        )
