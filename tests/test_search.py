"""
Tests of the draft search (src/draft_engine/search.py) that do NOT depend on the
trained model: the "model" here is an artificial evaluation function whose every
value is known. That separates two questions the real model mixes together:

    is the search algorithm correct?   <- answered here, immediately, at any data level
    is the win-probability model good? <- answered by `python -m src.tune` / real data

Ground truth comes from an independent exhaustive minimax (`exhaustive_value`
below): no shortlist, no alpha-beta pruning, no caching, just recursion over every
legal move to the end of the draft. With the shortlist widths set to the whole pool
and the depth set to the whole draft, the search must reproduce its values exactly -
which is what makes pruning, move ordering, the max/min perspective per team and the
denial-ban / lookahead-pick behaviour checkable without opinion.

    python -m unittest discover -s tests -t . -v
"""
from __future__ import annotations

import math
import random
import time
import unittest

from src.draft_engine.rules import DraftRules, Phase, Team
from src.draft_engine.search import recommend
from src.draft_engine.state import DraftState


class FixedModel:
    """P(team A wins) = sigmoid(score(A vs B) - score(B vs A)), where a team's score is
    the sum of its brawlers' `power`, plus `synergy` bonuses for teammate pairs and
    `counters` bonuses (x, y, bonus): the team holding x gets +bonus while y is on the
    other side. Works with partial teams. Antisymmetric by construction, so
    P(A vs B) + P(B vs A) = 1, exactly like the real model."""

    def __init__(self, power, synergy=(), counters=()):
        self.power = dict(power)
        self.synergy = [(frozenset(pair), bonus) for pair, bonus in synergy]
        self.counters = list(counters)
        self.calls = 0

    def _score(self, team, opponents):
        s = sum(self.power.get(b, 0.0) for b in team)
        s += sum(bonus for pair, bonus in self.synergy if pair <= set(team))
        s += sum(bonus for x, y, bonus in self.counters if x in team and y in opponents)
        return s

    def predict_proba(self, map_name, team_a, team_b):
        self.calls += 1
        return 1.0 / (1.0 + math.exp(-(self._score(team_a, team_b) - self._score(team_b, team_a))))


def exhaustive_value(model, state, pool):
    """Plain minimax over EVERY legal move to the end of the draft (the oracle).
    Returns P(A wins) under optimal play: team A maximizes it, team B minimizes it."""
    turn = state.current_turn()
    if turn is None:
        return model.predict_proba(state.map_name, state.picks[Team.A], state.picks[Team.B])
    values = [exhaustive_value(model, state.apply(c), pool) for c in state.available_brawlers(pool)]
    return max(values) if turn.team is Team.A else min(values)


def oracle_mover_values(model, state, pool):
    """{brawler: P(the mover's own team wins) after playing it, under optimal play}."""
    mover_is_a = state.current_turn().team is Team.A
    out = {}
    for c in state.available_brawlers(pool):
        v = exhaustive_value(model, state.apply(c), pool)
        out[c] = v if mover_is_a else 1.0 - v
    return out


# A small but complete draft - 1 ban each, 2 picks each in 1-2-1 blocks - so every
# ordering can be enumerated (7 brawlers: 5,040 complete drafts) while still having
# both ban and pick phases and the same block structure as the real 3-ban, 1-2-2-1 draft.
def small_rules(pick_first=Team.A, ban_first=Team.A):
    return DraftRules(
        bans_per_team=1, picks_per_team=2, pick_pattern=(1, 2, 1), pick_first=pick_first, ban_first=ban_first,
    )


def full_search(model, state, pool, **kw):
    """recommend() configured to be exact: whole-draft depth, no shortlisting, no time limit."""
    n = len(pool)
    return recommend(
        model, state, pool, depth=state.rules.total_turns(), top_n=n, inner_top_n=n,
        num_alternatives=n, time_budget=None, **kw,
    )


def random_model(rng, pool):
    power = {b: rng.uniform(-1.5, 1.5) for b in pool}
    synergy = [(rng.sample(pool, 2), rng.uniform(-1.0, 1.0)) for _ in range(2)]
    counters = [(*rng.sample(pool, 2), rng.uniform(0.5, 2.5)) for _ in range(3)]
    return FixedModel(power, synergy, counters)


POOL7 = list("ABCDEFG")


class SearchMatchesExhaustiveMinimax(unittest.TestCase):
    """The core correctness check: on random evaluators the search's value for every
    candidate move equals exhaustive minimax, from either team's point of view."""

    def test_every_candidate_value_equals_the_oracle(self):
        for seed in range(4):
            for pick_first in (Team.A, Team.B):
                model = random_model(random.Random(seed), POOL7)
                state = DraftState(map_name=None, rules=small_rules(pick_first))
                expected = oracle_mover_values(model, state, POOL7)
                recs = full_search(model, state, POOL7)
                self.assertEqual({r.brawler for r in recs}, set(POOL7))
                for r in recs:
                    self.assertAlmostEqual(
                        r.win_probability, expected[r.brawler], places=9,
                        msg=f"seed={seed} pick_first={pick_first.name} move={r.brawler}",
                    )
                self.assertEqual(recs[0].win_probability, max(expected.values()))

    def test_mid_draft_states_for_both_teams(self):
        """Same check when it is B's turn, and in the pick phase (max vs min perspective)."""
        model = random_model(random.Random(11), POOL7)
        rules = small_rules()
        state = DraftState(map_name=None, rules=rules)
        for _ in range(rules.total_turns() - 2):  # walk to the last two plies
            turn = state.current_turn()
            expected = oracle_mover_values(model, state, POOL7)
            recs = full_search(model, state, POOL7)
            for r in recs:
                self.assertAlmostEqual(r.win_probability, expected[r.brawler], places=9, msg=str(turn))
            state = state.apply(recs[0].brawler)


class DenialBans(unittest.TestCase):
    def test_bans_the_brawler_the_opponent_would_otherwise_take(self):
        """The enemy has first pick and WALL is by far the strongest brawler, so it would
        take WALL first thing - our only way to stop that is to ban it (denial, Abschnitt 1a).
        Our own picks don't change at all by that ban; only the opponent's options do."""
        power = {"WALL": 3.0, "A": 0.5, "B": 0.4, "C": 0.3, "D": 0.2, "E": 0.1, "F": 0.0, "G": -0.1}
        model = FixedModel(power)
        state = DraftState(map_name=None, rules=small_rules(pick_first=Team.B))
        recs = full_search(model, state, list(power))
        self.assertEqual(recs[0].brawler, "WALL")
        self.assertGreater(recs[0].win_probability, recs[1].win_probability + 0.05)  # clearly, not by a hair

    def test_ban_choice_is_not_just_the_strongest_brawler(self):
        """A weak brawler that hard-counters our best pick is the right ban: banning the
        merely 'strong' one would leave our KING exposed."""
        power = {"KING": 2.0, "SLAYER": 0.0, "A": 0.3, "B": 0.2, "C": 0.1, "D": 0.0, "E": -0.1}
        counters = [("SLAYER", "KING", 4.0)]  # SLAYER's team gets +4 while KING is on the other team
        model = FixedModel(power, counters=counters)
        pool = list(power)
        state = DraftState(map_name=None, rules=small_rules(pick_first=Team.A))
        expected = oracle_mover_values(model, state, pool)
        recs = full_search(model, state, pool)
        best_value = max(expected.values())
        self.assertAlmostEqual(recs[0].win_probability, best_value, places=9)
        # whatever the oracle prefers, the search agrees with it move for move
        self.assertAlmostEqual(expected[recs[0].brawler], best_value, places=9)

    def test_bans_matter_even_when_the_search_is_cut_off_before_the_picks(self):
        """Regression: a depth-limited search used to score every ban identically (bans
        never entered the leaf evaluation), so all ban recommendations came out equal.
        Full real rules (3 bans each, 1-2-2-1), cut off at depth 1 = still inside the ban
        phase, which is where a live search sits at the first ban."""
        power = {f"B{i}": 1.0 - i * 0.15 for i in range(13)}
        power["WALL"] = 3.0  # the enemy picks first and would take it
        model = FixedModel(power)
        state = DraftState(map_name=None, rules=DraftRules(pick_first=Team.B))
        recs = recommend(model, state, list(power), depth=1, top_n=14, num_alternatives=6, time_budget=None)
        self.assertEqual(state.current_turn().phase, Phase.BAN)
        self.assertGreater(len({round(r.win_probability, 9) for r in recs}), 1)
        self.assertEqual(recs[0].brawler, "WALL")

    def test_the_cutoff_evaluation_follows_the_real_pick_order(self):
        """When the search stops before the picks, the leaf value fills the teams greedily -
        and it has to do so in the draft's real order, or the value of a ban is wrong.
        Hand derivation: strengths o1 > o2 > ... (B0 = o1, B3 = o4, ...). The enemy has first
        pick, so the order is enemy, us, us, enemy, enemy, us: we end up with the 2nd, 3rd and
        6th best remaining brawlers, the enemy with the 1st, 4th and 5th. Strengths are
        1.0, .85, .7, .55, .4, .25, .1 for o1..o7, so
            ban o1: (o3+o4+o7) - (o2+o5+o6) = 1.35 - 1.50 = -0.15
            ban o2:                          1.35 - 1.65 = -0.30
            ban o3, o5, o7 or below:         1.65 - 1.80 = -0.15
            ban o4 (B3):    (o2+o3+o7) - (o1+o5+o6) = 1.65 - 1.65 =  0.00   <- the unique best
            ban o6:                          1.65 - 1.95 = -0.30
        i.e. the ban that helps most is the brawler the enemy would take in its double pick.
        (The previous strictly alternating fill, with us first, recommended B1 here.)"""
        power = {f"B{i}": [1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, -0.05, -0.2, -0.35, -0.5, -0.65, -0.8][i] for i in range(13)}
        model = FixedModel(power)
        state = DraftState(map_name=None, rules=DraftRules(pick_first=Team.B))
        recs = recommend(model, state, list(power), depth=1, top_n=13, num_alternatives=12, time_budget=None)
        self.assertEqual(recs[0].brawler, "B3")
        self.assertAlmostEqual(recs[0].win_probability, 0.5, places=9)
        by_name = {r.brawler: r.win_probability for r in recs}
        self.assertAlmostEqual(by_name["B0"], 1 / (1 + math.exp(0.15)), places=9)
        self.assertAlmostEqual(by_name["B1"], 1 / (1 + math.exp(0.30)), places=9)
        self.assertAlmostEqual(by_name["B5"], 1 / (1 + math.exp(0.30)), places=9)


class LookaheadPicks(unittest.TestCase):
    def setUp(self):
        # X looks best in isolation, but C (individually weak) hard-counters X.
        self.power = {"X": 1.0, "Y": 0.8, "C": 0.1, "F1": 0.0, "F2": 0.0, "F3": 0.0, "F4": 0.0}
        self.model = FixedModel(self.power, counters=[("C", "X", 3.0)])
        self.pool = list(self.power)
        rules = DraftRules(bans_per_team=1, picks_per_team=2, pick_pattern=(1, 2, 1), pick_first=Team.A)
        # bans already made (junk), so it is our first pick
        self.state = DraftState(
            map_name=None, rules=rules, bans={Team.A: ["F1"], Team.B: ["F2"]}, picks={Team.A: [], Team.B: []},
        )

    def test_the_myopic_pick_is_a_mistake_the_search_avoids(self):
        # myopic = best immediate win chance against an empty enemy team
        myopic = max(self.pool, key=lambda c: self.model.predict_proba(None, [c], []))
        self.assertEqual(myopic, "X")

        recs = full_search(self.model, self.state, self.pool)
        oracle = oracle_mover_values(self.model, self.state, self.pool)
        self.assertNotEqual(recs[0].brawler, myopic)  # counter-aware: X gets answered with C
        self.assertAlmostEqual(recs[0].win_probability, max(oracle.values()), places=9)
        # and the search knows what the myopic pick is really worth once the enemy answers
        x = next(r for r in recs if r.brawler == myopic)
        self.assertLess(x.win_probability, self.model.predict_proba(None, [myopic], []) - 0.2)

    def test_shallow_search_is_still_a_sane_answer(self):
        """depth=1 (the enemy's answer only via the greedy fill) must already see the counter."""
        recs = recommend(self.model, self.state, self.pool, depth=1, top_n=7, num_alternatives=6, time_budget=None)
        by_name = {r.brawler: r.win_probability for r in recs}
        self.assertLess(by_name["X"], by_name["Y"])


class RolesAndBookkeeping(unittest.TestCase):
    def test_swapping_the_two_teams_swaps_nothing_but_the_labels(self):
        """Team B moving first in a mirrored setup must get the same answer as team A does:
        checks that the maximize (A) / minimize (B) perspective is handled symmetrically."""
        model = random_model(random.Random(5), POOL7)
        mirrored = FixedModel(
            model.power, [(tuple(p), b) for p, b in model.synergy], model.counters
        )
        a_first = DraftState(map_name=None, rules=small_rules(pick_first=Team.A, ban_first=Team.A))
        b_first = DraftState(map_name=None, rules=small_rules(pick_first=Team.B, ban_first=Team.B))
        self.assertIs(a_first.current_turn().team, Team.A)
        self.assertIs(b_first.current_turn().team, Team.B)
        recs_a = full_search(model, a_first, POOL7)
        recs_b = full_search(mirrored, b_first, POOL7)
        self.assertEqual([r.brawler for r in recs_a], [r.brawler for r in recs_b])
        for ra, rb in zip(recs_a, recs_b):
            self.assertAlmostEqual(ra.win_probability, rb.win_probability, places=9)

    def test_never_recommends_a_brawler_that_is_already_used(self):
        rng = random.Random(3)
        pool = [f"B{i}" for i in range(15)]
        model = random_model(rng, pool)
        state = DraftState(map_name=None, rules=DraftRules())
        for _ in range(DraftRules().total_turns()):
            turn = state.current_turn()
            recs = recommend(model, state, pool, depth=2, top_n=8, num_alternatives=3, time_budget=None)
            self.assertTrue(recs, f"no recommendation at ply {state.ply} ({turn})")
            self.assertFalse({r.brawler for r in recs} & state.used_brawlers())
            self.assertEqual(len({r.brawler for r in recs}), len(recs))
            state = state.apply(recs[0].brawler)
        self.assertIsNone(state.current_turn())
        self.assertEqual(recommend(model, state, pool), [])  # nothing to recommend once complete

    def test_delta_is_relative_to_the_movers_win_chance_before_the_move(self):
        model = FixedModel({"A": 1.0, "B": 0.5, "C": 0.0, "D": -0.5, "E": -1.0, "F": 0.2, "G": 0.1})
        state = DraftState(map_name=None, rules=small_rules())
        recs = full_search(model, state, list("ABCDEFG"))
        # every candidate's delta is its value minus one shared baseline
        baselines = {round(r.win_probability - r.delta, 9) for r in recs}
        self.assertEqual(len(baselines), 1)

    def test_batched_and_unbatched_models_agree(self):
        """The search uses predict_proba_many when the model has it; both paths must give the same numbers."""
        class Batched(FixedModel):
            def predict_proba_many(self, map_name, situations):
                return [self.predict_proba(map_name, a, b) for a, b in situations]

        rng = random.Random(9)
        plain = random_model(rng, POOL7)
        batched = Batched(plain.power, [(tuple(p), b) for p, b in plain.synergy], plain.counters)
        state = DraftState(map_name=None, rules=small_rules())
        self.assertEqual(
            [(r.brawler, round(r.win_probability, 12)) for r in full_search(plain, state, POOL7)],
            [(r.brawler, round(r.win_probability, 12)) for r in full_search(batched, state, POOL7)],
        )


class TimeBudget(unittest.TestCase):
    def test_a_slow_model_still_answers_within_the_budget_using_a_shallower_search(self):
        class Slow(FixedModel):
            def predict_proba(self, map_name, team_a, team_b):
                time.sleep(0.0005)
                return super().predict_proba(map_name, team_a, team_b)

        pool = [f"B{i}" for i in range(12)]
        model = Slow({b: 1.0 - i * 0.1 for i, b in enumerate(pool)})
        state = DraftState(map_name=None, rules=DraftRules())
        stats: dict = {}
        started = time.perf_counter()
        recs = recommend(model, state, pool, depth=6, top_n=12, num_alternatives=3, time_budget=0.05, stats=stats)
        elapsed = time.perf_counter() - started
        self.assertTrue(recs)
        self.assertLess(stats["depth"], 6)   # the deeper iterations were abandoned...
        self.assertGreaterEqual(stats["depth"], 1)  # ...but depth 1 always completes
        self.assertLess(elapsed, 5.0)        # and the answer came back in bounded time


if __name__ == "__main__":
    unittest.main()
