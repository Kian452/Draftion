"""
Runner for a hosted deployment (Railway): serves the web app and, when a Brawl
Stars API key is configured, keeps the data and the model fresh in the same process.

Why one process: a Railway volume can be attached to a single service only, so
a separate crawler/cron service could not share the SQLite database and the trained
models with the web service. Everything therefore runs here:

  - a background thread crawls every CRAWL_INTERVAL_HOURS (the first run is a
    broad discovery, later runs re-poll the known high-rank pool, exactly like
    `python -m src.crawler --reseed-from-rank ...`),
  - once at least MIN_TRAIN_ROWS usable battles exist it retrains every
    RETRAIN_INTERVAL_HOURS - but only if RETRAIN_MIN_NEW_BATTLES new usable battles
    arrived since the last model - and writes the result as a NEW versioned file
    (model.save_versioned; the newest KEEP_MODELS are kept),
  - after every crawl and every retrain the API re-selects its model (newest trained
    version, or the raw-stats fallback while there isn't enough data yet),
  - uvicorn serves the API + frontend on $PORT.

Without BS_API_TOKEN (or with DRAFTION_WORKER=0) only the web app runs, which is
what the first deployment does. All settings are environment variables, listed in
config.py.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable

from dotenv import load_dotenv

from .config import Settings

log = logging.getLogger("serve")


def crawl_once(s: Settings) -> None:
    from .crawler import crawl

    crawl(
        s.db_path,
        countries=["global"],
        seed_limit=200,
        max_players=s.crawl_max_players,
        # Don't re-poll a player more often than the crawl interval; at least an hour.
        recrawl_hours=max(1, int(s.crawl_interval_hours)),
        reseed_from_rank=s.min_rank,
        reseed_battle_type=s.ranked_type,
    )


def retrain_if_ready(s: Settings, reload_model: Callable[[], None]) -> bool:
    """Train on the current database if there is enough new data. Returns True if a
    new model version was written."""
    from .brawlers import load_brawlers
    from .extract import build_training_table
    from .model import WinProbabilityModel, latest_version, save_versioned, train_model

    usable = len(build_training_table(s.db_path, s.ranked_type, s.min_rank))
    if usable < s.min_train_rows:
        log.info("Not training yet: %d usable battles (type=%s, rank>=%d), need %d.",
                 usable, s.ranked_type, s.min_rank, s.min_train_rows)
        return False
    latest = latest_version(s.model_dir)
    if latest is not None:
        trained_on = getattr(WinProbabilityModel.load(str(latest)), "n_battles", 0)
        if usable - trained_on < s.retrain_min_new_battles:
            log.info("Not retraining: only %d new usable battles since the last model (%d), need %d.",
                     usable - trained_on, trained_on, s.retrain_min_new_battles)
            return False
    model, _ = train_model(
        s.db_path, s.ranked_type, s.min_rank,
        backend=s.backend, brawlers=load_brawlers(s.brawlers_path),
    )
    path = save_versioned(model, s.model_dir, s.keep_models)
    reload_model()
    log.info("Retrained on %d battles -> %s (held-out %s).", usable, path, model.metrics or "n/a")
    return True


def worker_loop(s: Settings, reload_model: Callable[[], None], stop: threading.Event) -> None:
    last_train = float("-inf")
    while not stop.is_set():
        started = time.monotonic()
        try:
            log.info("Crawl run starting (up to %d players).", s.crawl_max_players)
            crawl_once(s)
        except Exception:
            # e.g. 403 = the key's allowed IP doesn't match; keep the web app up and retry next cycle
            log.exception("Crawl run failed - will retry next cycle.")
        if time.monotonic() - last_train >= s.retrain_interval_hours * 3600:
            try:
                retrain_if_ready(s, reload_model)
                last_train = time.monotonic()  # also when skipped for lack of new data: ask again next interval
            except Exception:
                log.exception("Retraining failed - keeping the current model.")
        try:
            reload_model()  # new battles -> the raw-stats fallback (if it is what's served) follows them
        except Exception:
            log.exception("Re-selecting the model failed - keeping the current one.")
        wait = max(60.0, s.crawl_interval_hours * 3600 - (time.monotonic() - started))
        stop.wait(wait)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()  # local runs: BS_API_TOKEN etc. from .env; on Railway they come from the service variables

    import uvicorn

    from . import api

    if os.environ.get("BS_API_TOKEN") and os.environ.get("DRAFTION_WORKER", "1") != "0":
        s = Settings.from_env()
        threading.Thread(
            target=worker_loop, args=(s, api.reload_model, threading.Event()),
            daemon=True, name="worker",
        ).start()
        log.info("Background crawler + retrainer on (every %sh, type=%s, min rank %d).",
                 s.crawl_interval_hours, s.ranked_type, s.min_rank)
    else:
        log.info("Background crawler off (no BS_API_TOKEN, or DRAFTION_WORKER=0) - serving the web app only.")

    uvicorn.run(api.app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
