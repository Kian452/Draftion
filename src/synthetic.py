"""
Synthetic data generator + end-to-end smoke test.

There's no real crawled data yet (data/brawl.db doesn't exist until the
crawler has run against a live API key for a while). This module generates
a fake-but-structurally-real Ranked dataset - going through the SAME
store_battle()/_resolve_winner_team() code path the real crawler uses, not
a shortcut - with a known, hand-picked bias (some brawlers stronger, a
couple of synergy pairs, a couple of hard counters) baked into the win
probabilities. That makes it useful for more than "does it crash": after
training, the model's calibration report should reflect that structure, and
the draft search should visibly avoid the hard-countered brawler / lean into
the synergy pair.

Not meant to ship or to stand in for real data - purely a pipeline
correctness check for Abschnitt 8/9/10 while waiting on Abschnitt 4 to
accumulate real Ranked matches.

Usage:
    python -m src.synthetic run
"""
from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta
from pathlib import Path

from .db import connect, store_battle

log = logging.getLogger("synthetic")

SMOKE_DIR = "data/_smoke"
SMOKE_DB = f"{SMOKE_DIR}/brawl.db"
SMOKE_MODEL = f"{SMOKE_DIR}/win_model.joblib"
SMOKE_BRAWLERS = f"{SMOKE_DIR}/brawlers.json"

MAPS = ["Synthetic Arena", "Synthetic Docks", "Synthetic Ruins"]
BATTLE_TYPE = "soloRanked"  # what real Ranked draft battles are called in the battlelog
MIN_RANK = 10  # arbitrary stand-in cutoff; real soloRanked players show tiers 4..18


def _build_bias(rng: random.Random, brawlers: list[str]) -> dict:
    strength = {b: rng.uniform(-1.0, 1.0) for b in brawlers}
    # A couple of explicit synergy/counter relationships so calibration and
    # the draft search have concrete structure to recover.
    synergy_pair = tuple(rng.sample(brawlers, 2))
    counter_pair = tuple(rng.sample(brawlers, 2))  # counter_pair[0] hard-counters counter_pair[1]
    return {
        "strength": strength,
        "synergy_pair": synergy_pair,
        "synergy_bonus": 0.6,
        "counter_pair": counter_pair,
        "counter_bonus": 1.2,
    }


def _team_score(team: list[str], opponents: list[str], bias: dict) -> float:
    score = sum(bias["strength"][b] for b in team)
    if bias["synergy_pair"][0] in team and bias["synergy_pair"][1] in team:
        score += bias["synergy_bonus"]
    if bias["counter_pair"][0] in team and bias["counter_pair"][1] in opponents:
        score += bias["counter_bonus"]
    return score


def _sigmoid(x: float) -> float:
    import math
    return 1.0 / (1.0 + math.exp(-x))


def generate_battle_entry(rng: random.Random, brawlers: list[str], bias: dict, battle_time: datetime) -> tuple[dict, str]:
    six = rng.sample(brawlers, 6)
    team0, team1 = six[:3], six[3:]
    score0 = _team_score(team0, team1, bias)
    score1 = _team_score(team1, team0, bias)
    p_team0_wins = _sigmoid((score0 - score1) + rng.gauss(0, 0.4))
    winner_idx = 0 if rng.random() < p_team0_wins else 1

    # Like the real API: no `battle.rank` in soloRanked - every player's brawler
    # carries their Ranked tier in `brawler.trophies`, and db.battle_rank() derives
    # the lobby rank from those (so the smoke test exercises that path).
    lobby_rank = MIN_RANK + rng.randint(0, 5)

    def make_player(i: int, team_idx: int, brawler: str) -> dict:
        tag = f"#SYN{team_idx}{i}{rng.randint(0, 999999):06d}"
        return {"tag": tag, "name": f"Bot{team_idx}{i}", "brawler": {
            "id": 16000000 + brawlers.index(brawler), "name": brawler, "power": 11,
            "trophies": lobby_rank + rng.randint(-1, 1),
        }}

    teams_players = [
        [make_player(i, t_idx, b) for i, b in enumerate(team)]
        for t_idx, team in enumerate((team0, team1))
    ]

    queried_team_idx = rng.randint(0, 1)
    queried_tag = teams_players[queried_team_idx][0]["tag"]
    result = "victory" if queried_team_idx == winner_idx else "defeat"

    entry = {
        "battleTime": battle_time.strftime("%Y%m%dT%H%M%S.000Z"),
        "event": {"map": rng.choice(MAPS)},
        "battle": {
            "mode": BATTLE_TYPE,
            "type": BATTLE_TYPE,
            "result": result,
            "duration": rng.randint(90, 180),
            "teams": teams_players,
        },
    }
    return entry, queried_tag


def generate_dataset(db_path: str, brawlers: list[str], n_battles: int = 4000, seed: int = 42) -> dict:
    rng = random.Random(seed)
    bias = _build_bias(rng, brawlers)

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(db_path).exists():
        Path(db_path).unlink()  # smoke-test DB only; always start fresh
    conn = connect(db_path)

    start = datetime(2026, 1, 1)
    inserted = 0
    for i in range(n_battles):
        entry, queried_tag = generate_battle_entry(rng, brawlers, bias, start + timedelta(minutes=i))
        if store_battle(conn, entry, queried_tag=queried_tag):
            inserted += 1

    conn.close()  # a leaked handle keeps the file locked on Windows
    log.info("Generated %d synthetic battles (%d brawlers) into %s", inserted, len(brawlers), db_path)
    log.info(
        "Baked-in bias: synergy_pair=%s (+%.2f), counter_pair=%s->%s (+%.2f)",
        bias["synergy_pair"], bias["synergy_bonus"],
        bias["counter_pair"][0], bias["counter_pair"][1], bias["counter_bonus"],
    )
    return {"brawlers": brawlers, "bias": bias}


def run_smoke_test() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import json

    import time

    from .brawlers import load_brawler_details
    from .draft_engine.rules import DraftRules, Team
    from .draft_engine.search import recommend
    from .draft_engine.state import DraftState
    from .model import train_model

    log.info("=== 1/4: generating synthetic dataset ===")
    # Use whatever roster data/brawlers.json holds (full roster after
    # `python -m src.brawlers fetch`, else the short seed list) so the latency
    # numbers below reflect the pool size you'll actually run with.
    details = load_brawler_details()
    brawlers = [d["name"] for d in details]
    meta = generate_dataset(SMOKE_DB, brawlers)
    Path(SMOKE_BRAWLERS).write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("=== 2/4: training model (logreg) ===")
    model, df = train_model(
        SMOKE_DB, BATTLE_TYPE, MIN_RANK, backend="logreg", brawlers=brawlers, data_source="synthetic",
    )
    model.save(SMOKE_MODEL)
    log.info("Training rows: %d", len(df))

    # The documented way to train is the CLI, and a model it writes must be loadable by
    # another process (the API server). Pickling a class defined in `__main__` breaks
    # that, which importing train_model() in-process (as above) can never reveal.
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cli_model = f"{tmp}/cli_model.joblib"
        subprocess.run(
            [sys.executable, "-m", "src.model", "train", "--db", SMOKE_DB, "--battle-type", BATTLE_TYPE,
             "--min-rank", str(MIN_RANK), "--out", cli_model, "--brawlers", SMOKE_BRAWLERS],
            check=True, capture_output=True,
        )
        from .model import WinProbabilityModel, latest_version
        assert WinProbabilityModel.load(cli_model).predict_proba(None, brawlers[0:3], brawlers[3:6]) > 0

        # Without --out the CLI writes a NEW versioned file (what the running API picks up).
        cmd = [sys.executable, "-m", "src.model", "train", "--db", SMOKE_DB, "--battle-type", BATTLE_TYPE,
               "--min-rank", str(MIN_RANK), "--model-dir", f"{tmp}/models", "--keep", "2", "--brawlers", SMOKE_BRAWLERS]
        for _ in range(2):
            subprocess.run(cmd, check=True, capture_output=True)
            time.sleep(1.1)  # version names are stamped to the second
        subprocess.run(cmd, check=True, capture_output=True)
        from .model import list_versions
        assert len(list_versions(f"{tmp}/models")) == 2, list_versions(f"{tmp}/models")  # pruned to --keep
        newest = WinProbabilityModel.load(str(latest_version(f"{tmp}/models")))
        assert newest.metrics.get("c") and newest.trained_at, newest.metrics
    log.info("CLI round trip OK (models trained by `python -m src.model train` load in another process; versions are pruned)")

    log.info("=== 3/4: sanity-checking recovered bias ===")
    bias = meta["bias"]
    # The model must not care which side a team is on: P(X beats Y) + P(Y beats X) = 1.
    for x, y in ((brawlers[0:3], brawlers[3:6]), (brawlers[10:13], brawlers[20:23])):
        total = model.predict_proba(None, x, y) + model.predict_proba(None, y, x)
        assert abs(total - 1.0) < 0.02, f"model is side-dependent: P(A)+P(B) = {total:.3f} for {x} vs {y}"
    log.info("Side symmetry OK (P(A)+P(B) within 0.02 of 1)")
    a, b = bias["counter_pair"]
    p_with_counter = model.predict_proba(None, [a], [b])
    p_without = model.predict_proba(None, [], [b])
    log.info(
        "P(win | %s vs %s) = %.3f  vs  P(win | nobody vs %s) = %.3f "
        "(expect the counter pick to score higher)",
        a, b, p_with_counter, b, p_without,
    )

    log.info("=== 4/4: running draft search + FastAPI /recommend end-to-end ===")
    state = DraftState(map_name=None, rules=DraftRules())
    turn = state.current_turn()
    recs = recommend(model, state, brawlers, depth=4, top_n=16, num_alternatives=7, time_budget=1.0)
    log.info("First recommended action (%s, Team %s): %s", turn.phase.value, turn.team.name, recs[:3])

    # Latency at the draft stages a user actually waits on, using the same
    # settings the API uses (see DraftPredictor). Bans/picks here are just the
    # first N pool entries - only the stage matters, not which brawlers.
    for label, n_used in (("first ban", 0), ("mid bans", 3), ("first pick", 6), ("last pick", 11)):
        s = DraftState(map_name=None, rules=DraftRules())
        for b in brawlers[:n_used]:
            s = s.apply(b)
        stats: dict = {}
        t0 = time.perf_counter()
        recommend(model, s, brawlers, depth=4, top_n=16, num_alternatives=7, time_budget=1.0, stats=stats)
        log.info(
            "recommend() latency at %-10s = %.2fs, searched %d plies, %d model evals (pool=%d brawlers)",
            label, time.perf_counter() - t0, stats["depth"], stats["model_calls"], len(brawlers),
        )

    _run_api_smoke_test()

    log.info("Smoke test OK. Synthetic artifacts under %s/ (safe to delete).", SMOKE_DIR)


def _run_api_smoke_test() -> None:
    import os

    os.environ["WIN_MODEL_PATH"] = SMOKE_MODEL
    os.environ["BRAWLERS_PATH"] = SMOKE_BRAWLERS
    os.environ["DB_PATH"] = SMOKE_DB
    os.environ["BATTLE_TYPE"] = BATTLE_TYPE

    from fastapi.testclient import TestClient

    from .api import app

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["model_loaded"], r.text
        # this model was trained on synthetic data and pinned via WIN_MODEL_PATH: served as-is, flagged as a demo
        assert r.json()["model_kind"] == "trained" and r.json()["model_confidence"] == "demo", r.json()
        assert r.json()["model_metrics"]["n_test"] > 0, r.json()
        r = client.get("/brawlers")
        assert r.status_code == 200 and len(r.json()) > 0 and "name" in r.json()[0], r.text
        r = client.get("/rules")
        rules = r.json()
        assert r.status_code == 200 and len(rules["turns"]) == 2 * (rules["bans_per_team"] + rules["picks_per_team"]), r.text
        rules_b = client.get("/rules", params={"first_pick": "B"}).json()
        first_pick_turn = rules["bans_per_team"] * 2  # index of the first pick turn
        assert rules["turns"][first_pick_turn]["team"] == "A" and rules_b["turns"][first_pick_turn]["team"] == "B", (rules, rules_b)
        assert client.get("/rules", params={"first_pick": "C"}).status_code == 422
        team_a, team_b = ([b["name"] for b in client.get("/brawlers").json()][i:i + 3] for i in (0, 3))
        r = client.post("/analyze", json={"picks_a": team_a, "picks_b": team_b})
        an = r.json()
        assert r.status_code == 200 and 0 < an["win_probability_a"] < 1, r.text
        assert len(an["matchups"]) == 3 and all(len(row) == 3 and all(0 < p < 1 for p in row) for row in an["matchups"]), an
        assert len(an["impact_a"]) == len(an["impact_b"]) == 3 and len(an["synergy_a"]) == len(an["synergy_b"]) == 3, an
        # swapping the teams must give the complementary probabilities everywhere
        sw = client.post("/analyze", json={"picks_a": team_b, "picks_b": team_a}).json()
        assert abs(an["matchups"][0][1] + sw["matchups"][1][0] - 1.0) < 1e-9, (an["matchups"], sw["matchups"])
        r = client.get("/maps")
        assert r.status_code == 200, r.text
        assert all("active" in m for m in r.json()), r.json()[:2]
        r = client.post("/recommend", json={"bans_a": [], "bans_b": [], "picks_a": [], "picks_b": []})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["phase"] == "ban" and body["team_on_clock"] == "A", body
        assert len(body["recommendations"]) > 0, body
        log.info("API smoke test OK: %s", body["recommendations"][0])


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run_smoke_test()
    else:
        print("Usage: python -m src.synthetic run")
