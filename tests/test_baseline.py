"""The raw-stats fallback model (src/baseline.py) - what answers while there is too little data."""
from __future__ import annotations

import unittest

import pandas as pd

from src.baseline import RawStatsModel


def table(rows):
    """rows: (team_a, team_b, winner) -> the training-table shape extract.build_training_table produces."""
    return pd.DataFrame(
        [{"team_a": a, "team_b": b, "winner": w, "weight": 1.0, "map": None, "rank": 12, "battle_time": f"{i:08d}"}
         for i, (a, b, w) in enumerate(rows)]
    )


def strong_vs_weak(n_battles=100, strong_wins=0.8):
    """STRONG wins `strong_wins` of its games against WEAK; the other slots are neutral filler."""
    rows = []
    for i in range(n_battles):
        won = i < n_battles * strong_wins
        rows.append((["STRONG", "F1", "F2"], ["WEAK", "F3", "F4"], 1 if won else 0))
    return table(rows)


class RawStats(unittest.TestCase):
    def test_a_brawler_that_wins_more_gets_the_higher_strength(self):
        model = RawStatsModel.from_table(strong_vs_weak())
        self.assertGreater(model.strength["STRONG"], 0)
        self.assertLess(model.strength["WEAK"], 0)
        self.assertGreater(model.predict_proba(None, ["STRONG"], ["WEAK"]), 0.5)
        self.assertLess(model.predict_proba(None, ["WEAK"], ["STRONG"]), 0.5)

    def test_it_is_side_symmetric(self):
        model = RawStatsModel.from_table(strong_vs_weak())
        for a, b in ((["STRONG", "F1"], ["WEAK"]), (["F1"], ["F3", "WEAK"]), (["STRONG"], ["STRONG"])):
            self.assertAlmostEqual(model.predict_proba(None, a, b) + model.predict_proba(None, b, a), 1.0, places=12)
        self.assertAlmostEqual(model.predict_proba(None, ["STRONG"], ["STRONG"]), 0.5, places=12)

    def test_partial_and_unknown_teams_are_fine(self):
        model = RawStatsModel.from_table(strong_vs_weak())
        self.assertEqual(model.predict_proba(None, [], []), 0.5)
        self.assertEqual(model.predict_proba(None, ["NEVER SEEN"], ["ALSO UNSEEN"]), 0.5)
        self.assertGreater(model.predict_proba(None, ["STRONG"], []), 0.5)
        self.assertGreater(model.predict_proba(None, ["STRONG", None, ""], ["F3"]), 0.5)  # empty slots are ignored

    def test_few_games_are_shrunk_hard_towards_even(self):
        """A brawler that won its 2 games is not treated like one that won 80 of 100."""
        rows = [(["LUCKY", "F1", "F2"], ["F3", "F4", "F5"], 1)] * 2
        rows += [(["PROVEN", "F1", "F2"], ["F3", "F4", "F5"], 1 if i < 80 else 0) for i in range(100)]
        model = RawStatsModel.from_table(table(rows))
        self.assertGreater(model.strength["LUCKY"], 0)
        self.assertLess(model.strength["LUCKY"], model.strength["PROVEN"] / 2)

    def test_an_empty_table_gives_a_neutral_model(self):
        model = RawStatsModel.from_table(table([]))
        self.assertEqual(model.n_battles, 0)
        self.assertEqual(model.predict_proba(None, ["X"], ["Y"]), 0.5)

    def test_it_matches_the_interface_the_search_uses(self):
        model = RawStatsModel.from_table(strong_vs_weak())
        many = model.predict_proba_many("any map", [(["STRONG"], ["WEAK"]), (["WEAK"], ["STRONG"])])
        self.assertEqual(len(many), 2)
        self.assertAlmostEqual(many[0] + many[1], 1.0, places=12)
        self.assertEqual(model.kind, "raw-stats")
        self.assertIn("STRONG", model.known_brawlers())

    def test_the_draft_search_runs_on_it(self):
        from src.draft_engine.rules import DraftRules
        from src.draft_engine.search import recommend
        from src.draft_engine.state import DraftState

        model = RawStatsModel.from_table(strong_vs_weak())
        pool = ["STRONG", "WEAK", "F1", "F2", "F3", "F4", "N1", "N2", "N3", "N4", "N5", "N6", "N7"]
        recs = recommend(model, DraftState(map_name=None, rules=DraftRules()), pool, depth=2, top_n=8, time_budget=None)
        self.assertTrue(recs)


if __name__ == "__main__":
    unittest.main()
