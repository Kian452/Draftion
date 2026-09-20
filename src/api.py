"""
FastAPI backend (Abschnitt 10).

POST /recommend takes the current draft state (map, bans/picks so far for
both teams) and returns whose turn it is plus a ranked list of
recommendations - built from the draft search engine (Abschnitt 9) over the
win-probability model (Abschnitt 8). Model and brawler roster are loaded
once at startup (see predictor.py), not per request; a newer model version
written to WIN_MODEL_DIR (by the hosted worker, a cron job or a manual
`python -m src.model train`) is picked up without a restart (_maybe_reload).
Which model answers - trained, raw-stats fallback or demo - is decided in
model_select.py.

Run with:
    uvicorn src.api:app --reload
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .brawlers import load_brawler_details
from .config import Settings
from .db import active_ranked_maps, connect, distinct_maps
from .draft_engine.rules import DraftRules, Team
from .maps import load_map_details
from .model import resolve_model_path
from .model_select import confidence, select_model
from .predictor import DraftPredictor

log = logging.getLogger("api")

_predictor: Optional[DraftPredictor] = None
_trained_path: Optional[str] = None  # the trained-model file the current predictor was chosen against
_last_model_check = 0.0
_reload_lock = threading.Lock()
_brawler_details: list[dict] = []
_rules = DraftRules()  # single source of truth for the ban/pick sequence: predictor and /rules both read this


def _cfg() -> Settings:
    # Read fresh: cheap, and lets a test set the environment after importing this module.
    return Settings.from_env()


def _build_predictor() -> tuple[Optional[DraftPredictor], Optional[str]]:
    s = _cfg()
    selection = select_model(s)
    if selection.model is None:
        return None, selection.trained_path
    predictor = DraftPredictor(
        selection.model, brawlers_path=s.brawlers_path, rules=_rules, time_budget=s.draft_time_budget,
    )
    return predictor, selection.trained_path


def reload_model() -> None:
    """Re-select the model (newest trained version, else the raw-stats fallback, see
    model_select.py) and swap it in. Called by the hosted worker (src/serve.py) after
    each crawl and retrain; requests already running keep the old predictor."""
    global _predictor, _trained_path, _last_model_check
    with _reload_lock:
        _predictor, _trained_path = _build_predictor()
        _last_model_check = time.monotonic()


def _maybe_reload() -> None:
    """Pick up a model version written by ANOTHER process (a cron job, a manual
    `python -m src.model train`) without a restart. Throttled to one directory listing
    per MODEL_CHECK_SECONDS; a new file is the trigger, the reload itself is rare."""
    global _last_model_check
    s = _cfg()
    if s.model_path or time.monotonic() - _last_model_check < s.model_check_seconds:
        return  # an explicit WIN_MODEL_PATH pins the file
    _last_model_check = time.monotonic()
    if resolve_model_path(None, s.model_dir, s.legacy_model_path) != _trained_path:
        try:
            reload_model()
            log.info("Picked up a newer model version.")
        except Exception:
            log.exception("Reloading the model failed - keeping the current one.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor, _trained_path, _brawler_details, _last_model_check
    s = _cfg()
    # Loaded once here (not per request) and independent of the model, so the
    # pick/ban grid still works before a model has been trained.
    _brawler_details = load_brawler_details(s.brawlers_path)
    try:
        _predictor, _trained_path = _build_predictor()
    except FileNotFoundError:
        log.warning("The model file %s does not exist.", s.model_path)
        _predictor = None
    if _predictor is None:
        log.warning(
            "No model available - /recommend will return 503 until you run "
            "`python -m src.model train ...`. /brawlers and /maps still work."
        )
    _last_model_check = time.monotonic()
    yield


app = FastAPI(title="Brawl Stars Ranked Draft AI", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.middleware("http")
async def no_cache(request, call_next):
    """
    This is a single-service, actively-edited local tool (Abschnitt 0) - a
    browser silently serving a stale cached web/app.js after an update (own
    edits, or a future `git pull`) is a confusing failure mode with no
    obvious symptom pointing back to "your browser cached an old file".
    Not worth trading away for the static-asset caching a bigger/public
    deployment would want.
    """
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


class RecommendRequest(BaseModel):
    map: Optional[str] = None
    bans_a: list[str] = Field(default_factory=list)
    bans_b: list[str] = Field(default_factory=list)
    picks_a: list[str] = Field(default_factory=list)
    picks_b: list[str] = Field(default_factory=list)
    num_alternatives: int = Field(default=3, ge=0, le=15)  # each extra one is searched too - keep it bounded
    first_pick: Literal["A", "B"] = "A"  # which team has first pick this match (Team A = your team)


class RecommendationOut(BaseModel):
    brawler: str
    win_probability: float
    delta: float


class RecommendResponse(BaseModel):
    phase: Optional[str]
    team_on_clock: Optional[str]
    recommendations: list[RecommendationOut]


@app.get("/rules")
def get_rules(first_pick: Literal["A", "B"] = "A") -> dict:
    """The full ban/pick sequence, so the frontend can work out whose turn it is
    from the number of actions so far - instantly, without waiting for a
    (slower) /recommend round trip. Still defined only in draft_engine/rules.py.
    `first_pick` = which team has first pick this match (the pick order depends on it)."""
    rules = replace(_rules, pick_first=Team[first_pick])
    return {
        "bans_per_team": rules.bans_per_team,
        "picks_per_team": rules.picks_per_team,
        "first_pick": first_pick,
        "turns": [{"phase": t.phase.value, "team": t.team.name} for t in rules.turn_order()],
    }


@app.get("/health")
def health() -> dict:
    _maybe_reload()
    model = _predictor.model if _predictor else None
    loaded = model is not None
    return {
        "status": "ok",
        "model_loaded": loaded,
        # "trained" (logistic regression / gbm) or "raw-stats" (the preliminary fallback)
        "model_kind": getattr(model, "kind", "trained") if loaded else None,
        # demo | low | early | ok - how far the UI should trust the numbers (model_select.confidence)
        "model_confidence": confidence(model, _cfg()),
        # models pickled before this field existed were always trained on real data
        "model_data_source": getattr(model, "data_source", "real") if loaded else None,
        "model_training_battles": getattr(model, "n_battles", 0) if loaded else None,
        "model_trained_at": (getattr(model, "trained_at", "") or None) if loaded else None,
        # held-out test metrics of the last training run: auc, brier, logloss, n_test, c
        "model_metrics": (getattr(model, "metrics", None) or None) if loaded else None,
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    _maybe_reload()
    if _predictor is None:
        raise HTTPException(503, "No trained model loaded yet - run `python -m src.model train` first.")
    turn, recs = _predictor.recommend(
        req.map, req.bans_a, req.bans_b, req.picks_a, req.picks_b,
        num_alternatives=req.num_alternatives, first_pick=req.first_pick,
    )
    if turn is None:
        return RecommendResponse(phase=None, team_on_clock=None, recommendations=[])
    return RecommendResponse(
        phase=turn.phase.value,
        team_on_clock=turn.team.name,
        recommendations=[
            RecommendationOut(brawler=r.brawler, win_probability=r.win_probability, delta=r.delta)
            for r in recs
        ],
    )


class BrawlerOut(BaseModel):
    name: str
    id: Optional[int] = None
    image: Optional[str] = None


class AnalyzeRequest(BaseModel):
    map: Optional[str] = None
    # Team A = your team. Bounded so a bad client can't ask for a huge probe batch.
    picks_a: list[str] = Field(default_factory=list, max_length=5)
    picks_b: list[str] = Field(default_factory=list, max_length=5)


class ScoredBrawler(BaseModel):
    brawler: str
    value: float


class ScoredPair(BaseModel):
    pair: list[str]
    value: float


class AnalyzeResponse(BaseModel):
    team_a: list[str]
    team_b: list[str]
    win_probability_a: float
    matchups: list[list[float]]  # [i][j] = chance team_a[i] beats team_b[j]
    impact_a: list[ScoredBrawler]
    impact_b: list[ScoredBrawler]
    synergy_a: list[ScoredPair]
    synergy_b: list[ScoredPair]


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> dict:
    """Breakdown of a finished draft for the results screen: overall win chance,
    who beats whom, each brawler's impact and teammate synergies."""
    _maybe_reload()
    if _predictor is None:
        raise HTTPException(503, "No trained model loaded yet - run `python -m src.model train` first.")
    return _predictor.analyze(req.map, req.picks_a, req.picks_b)


@app.get("/brawlers", response_model=list[BrawlerOut])
def get_brawlers() -> list[dict]:
    return _brawler_details


class MapOut(BaseModel):
    name: str
    mode: Optional[str] = None
    image: Optional[str] = None
    active: bool = True  # in the current Ranked pool (played recently in crawled Ranked battles)


# Fewer active maps than this means the crawl hasn't seen enough of the pool yet (six
# modes -> a real pool has several maps per mode), so nothing is hidden.
MIN_POOL_MAPS = 6


@app.get("/maps", response_model=list[MapOut])
def get_maps() -> list[dict]:
    """The map list from data/maps.json (`python -m src.maps fetch`), reconciled
    with the map names seen in crawled Ranked battles. The battlelog's spelling
    is what the trained model knows, so it wins: Brawlify writes "Belles Rock"
    where the game says "Belle's Rock", "Out In The Open" for "Out in the Open".
    Maps the crawl saw that Brawlify doesn't list are added (with the mode from
    the battlelog). The frontend also accepts a free-text map name.

    Each map carries `active`: True if it is in the CURRENT Ranked pool, i.e. was
    played in the last MAP_WINDOW_DAYS of crawled Ranked battles (db.active_ranked_maps -
    GET /events/rotation was measured NOT to be that pool). The frontend offers only the
    active maps by default, with a switch for the full list."""
    s = _cfg()
    maps = load_map_details(s.maps_path)
    active: dict[str, int] = {}
    if os.path.exists(s.db_path):
        conn = connect(s.db_path)
        try:
            seen = distinct_maps(conn, s.ranked_type)
            active = active_ranked_maps(conn, s.ranked_type, s.map_window_days, s.map_min_battles)
        finally:
            conn.close()
        by_key = {_map_key(m["name"]): m for m in maps}
        for name, mode in seen:
            entry = by_key.get(_map_key(name))
            if entry is not None:
                entry["name"] = name
            else:
                entry = {"name": name, "mode": _mode_label(mode), "image": None}
                maps.append(entry)
                by_key[_map_key(name)] = entry
    active_keys = {_map_key(name) for name in active}
    pool_known = len(active_keys) >= MIN_POOL_MAPS
    for m in maps:
        m["active"] = (not pool_known) or _map_key(m["name"]) in active_keys
    return maps


def _map_key(name: str) -> str:
    """Spelling-insensitive key: letters and digits only, lower-cased."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _mode_label(mode: Optional[str]) -> Optional[str]:
    """The battlelog's camelCase mode ("gemGrab") as the display name ("Gem Grab")."""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", mode).title() if mode else None


# Serve the web/ frontend (Abschnitt 11) if present, so the whole app runs
# off a single `uvicorn src.api:app` command.
_static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
if os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="web")
