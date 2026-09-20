"""
Loads the trained model + brawler roster once (Abschnitt 10: "Modell und
Brawlerliste einmal beim Start laden, nicht pro Request"), then wraps the
draft search engine (draft_engine/search.py) for the API layer.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Optional

from .brawlers import load_brawlers
from .draft_engine.analysis import analyze_draft
from .draft_engine.rules import DraftRules, Team, Turn
from .draft_engine.search import Recommendation
from .draft_engine.search import recommend as search_recommend
from .draft_engine.state import DraftState

log = logging.getLogger("predictor")


class DraftPredictor:
    def __init__(
        self,
        model,  # a model.WinProbabilityModel or baseline.RawStatsModel (see model_select.py)
        brawlers_path: str = "data/brawlers.json",
        rules: Optional[DraftRules] = None,
        depth: int = 4,
        top_n: int = 16,
        inner_top_n: int = 8,
        time_budget: float = 1.0,
    ):
        self.model = model
        self.pool = load_brawlers(brawlers_path)
        self.rules = rules or DraftRules()
        self.depth = depth
        self.top_n = top_n
        self.inner_top_n = inner_top_n
        self.time_budget = time_budget
        log.info(
            "Loaded model (backend=%s) and %d-brawler pool from %s.",
            self.model.backend, len(self.pool), brawlers_path,
        )
        # Brawler names are the join key between the pool and the trained
        # model. If the roster's spelling ever differs from the battlelog
        # spelling the model was trained on (e.g. "Shelly" vs "SHELLY"), every
        # feature would silently miss and all brawlers would look identical.
        known = self.model.known_brawlers()
        unknown = [b for b in self.pool if b not in known]
        if unknown:
            log.warning(
                "%d of %d pool brawlers never appeared in the model's training data (e.g. %s). "
                "Expected for brand-new brawlers; if it's most of the pool, the names in %s "
                "don't match the names in the crawled battlelogs.",
                len(unknown), len(self.pool), unknown[:5], brawlers_path,
            )

    def analyze(self, map_name: Optional[str], picks_a: list[str], picks_b: list[str]) -> dict:
        """Post-draft breakdown for the results screen (see draft_engine/analysis.py)."""
        return analyze_draft(self.model, map_name, picks_a, picks_b)

    def recommend(
        self,
        map_name: Optional[str],
        bans_a: list[str],
        bans_b: list[str],
        picks_a: list[str],
        picks_b: list[str],
        num_alternatives: int = 3,
        first_pick: str = "A",
    ) -> tuple[Optional[Turn], list[Recommendation]]:
        # Which team has first pick is decided per match, so the pick order is
        # derived per request rather than fixed at startup.
        rules = replace(self.rules, pick_first=Team[first_pick])
        state = DraftState(
            map_name=map_name,
            rules=rules,
            bans={Team.A: list(bans_a), Team.B: list(bans_b)},
            picks={Team.A: list(picks_a), Team.B: list(picks_b)},
        )
        turn = state.current_turn()
        if turn is None:
            return None, []
        stats: dict = {}
        recs = search_recommend(
            self.model, state, self.pool,
            depth=self.depth, top_n=self.top_n, num_alternatives=num_alternatives,
            inner_top_n=self.inner_top_n, time_budget=self.time_budget, stats=stats,
        )
        log.info(
            "recommend: %s %s, searched %d plies in %.2fs (%d model evaluations)",
            turn.team.name, turn.phase.value, stats["depth"], stats["elapsed"], stats["model_calls"],
        )
        return turn, recs
