"""The ban/pick sequence (src/draft_engine/rules.py) and DraftState's bookkeeping."""
from __future__ import annotations

import unittest

from src.draft_engine.rules import DraftRules, Phase, Team
from src.draft_engine.state import DraftState

A, B = Team.A, Team.B


def sequence(rules):
    return [(t.phase.value, t.team.name) for t in rules.turn_order()]


class TurnOrder(unittest.TestCase):
    def test_bans_are_two_blocks_our_team_first(self):
        turns = sequence(DraftRules())
        self.assertEqual(turns[:6], [("ban", "A")] * 3 + [("ban", "B")] * 3)

    def test_picks_are_1_2_2_1_starting_with_the_team_that_has_first_pick(self):
        self.assertEqual(
            sequence(DraftRules(pick_first=A))[6:],
            [("pick", "A"), ("pick", "B"), ("pick", "B"), ("pick", "A"), ("pick", "A"), ("pick", "B")],
        )
        self.assertEqual(
            sequence(DraftRules(pick_first=B))[6:],
            [("pick", "B"), ("pick", "A"), ("pick", "A"), ("pick", "B"), ("pick", "B"), ("pick", "A")],
        )

    def test_first_pick_does_not_change_the_ban_phase(self):
        self.assertEqual(sequence(DraftRules(pick_first=A))[:6], sequence(DraftRules(pick_first=B))[:6])

    def test_every_team_gets_three_bans_and_three_picks(self):
        for first in (A, B):
            turns = DraftRules(pick_first=first).turn_order()
            self.assertEqual(len(turns), 12)
            for team in (A, B):
                self.assertEqual(sum(t.team is team and t.phase is Phase.BAN for t in turns), 3)
                self.assertEqual(sum(t.team is team and t.phase is Phase.PICK for t in turns), 3)

    def test_a_pattern_that_does_not_add_up_is_rejected(self):
        with self.assertRaises(ValueError):
            DraftRules(pick_pattern=(1, 1, 1))  # 2 and 1 picks, but picks_per_team is 3
        with self.assertRaises(ValueError):
            DraftRules(picks_per_team=4)  # the default 1-2-2-1 gives 3 each


class StateProgression(unittest.TestCase):
    def test_walking_the_whole_draft(self):
        state = DraftState(map_name=None, rules=DraftRules(pick_first=B))
        names = [f"B{i}" for i in range(12)]
        for i, name in enumerate(names):
            expected = DraftRules(pick_first=B).turn_order()[i]
            self.assertEqual(state.current_turn(), expected)
            state = state.apply(name)
        self.assertIsNone(state.current_turn())
        self.assertTrue(state.is_terminal())
        self.assertEqual(state.bans[A], names[0:3])
        self.assertEqual(state.bans[B], names[3:6])
        self.assertEqual(state.picks[B], [names[6], names[9], names[10]])
        self.assertEqual(state.picks[A], [names[7], names[8], names[11]])
        with self.assertRaises(ValueError):
            state.apply("ONE MORE")

    def test_apply_returns_a_new_state_and_leaves_the_old_one_alone(self):
        state = DraftState(map_name=None)
        after = state.apply("X")
        self.assertEqual(state.ply, 0)
        self.assertEqual(after.ply, 1)
        self.assertEqual(state.bans[A], [])
        self.assertEqual(after.available_brawlers(["X", "Y"]), ["Y"])


if __name__ == "__main__":
    unittest.main()
