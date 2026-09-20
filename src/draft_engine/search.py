"""
Draft search engine (Abschnitt 9) - the actual "AI bot".

Both requirements from Abschnitt 1 fall out of a single minimax formulation,
rather than needing separate denial/lookahead code paths:

  - value(state) := P(Team A wins), read from the win-probability model
    (Abschnitt 8) at a leaf/cutoff. Team A's turns pick the action that
    MAXIMIZES this value; Team B's turns pick the action that MINIMIZES it.
  - On Team A's ban turn, maximizing the resulting value means choosing the
    ban that most *denies* Team B's best achievable continuation -
    Abschnitt 1a's "denial" requirement.
  - On either team's pick turn, choosing the action that optimizes the
    value one or more plies ahead - after the opponent's best response -
    is exactly Abschnitt 1b's "lookahead" requirement (a pure greedy pick
    would instead only optimize the immediate leaf, ignoring the reply).

Practical constraints from Abschnitt 9 shape the implementation:
  - Full-width search is intractable (100+ candidates/turn x 12 plies), so
    every node first shortlists to the top `top_n` candidates by a cheap
    one-ply heuristic score, and only those get expanded. The root keeps a
    wide shortlist (it also produces the alternatives list); deeper nodes use
    a narrower `inner_top_n`.
  - At `depth` 0 (or a true terminal state) the model's own prediction is the
    value. That's why the model must support partial teams - it doubles as
    the static evaluator at a search cutoff.
  - Iterative deepening under a time budget: search 1 ply, then 2, 3, ...
    until `depth` or `time_budget` runs out, and answer with the last
    *completed* depth. Measured on a synthetic 109-brawler pool, a single
    fixed-depth-4 search took 2 s at the first ban but 145 s at the first
    pick - unusable for a live draft - whereas this keeps latency bounded
    regardless of pool size or phase, and simply searches deeper when the
    remaining tree is small.
  - Model calls are batched (`predict_proba_many`): scoring all candidates in
    one vectorized call is several times cheaper than one call each.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

from .rules import Phase, Team, Turn
from .state import DraftState

NEG_INF = float("-inf")
POS_INF = float("inf")

Situation = tuple[Sequence[str], Sequence[str]]  # (team_a picks, team_b picks)


class WinProbabilityModel(Protocol):
    def predict_proba(
        self, map_name: Optional[str], team_a: Sequence[str], team_b: Sequence[str]
    ) -> float: ...

    # Optional fast path - used when present, else predict_proba is looped:
    # def predict_proba_many(self, map_name, situations) -> list[float]


class _OutOfTime(Exception):
    """Raised inside a search iteration when the time budget is exhausted."""


class _CachedEvaluator:
    """
    Per-search wrapper around a WinProbabilityModel: memoizes results (the
    same (map, team_a, team_b) recurs across branches), batches cache misses
    into a single model call, and enforces the time budget.

    A fresh instance is created per recommend() call, never shared across
    requests.
    """

    def __init__(self, model: WinProbabilityModel):
        self._model = model
        self._cache: dict[tuple, float] = {}
        self.deadline: Optional[float] = None
        self.model_calls = 0  # situations actually sent to the model

    def check_time(self) -> None:
        if self.deadline is not None and time.perf_counter() > self.deadline:
            raise _OutOfTime

    def predict_proba(self, map_name: Optional[str], team_a: Sequence[str], team_b: Sequence[str]) -> float:
        return self.predict_many(map_name, [(team_a, team_b)])[0]

    def predict_many(self, map_name: Optional[str], situations: Sequence[Situation]) -> list[float]:
        keys = [(map_name, tuple(sorted(a)), tuple(sorted(b))) for a, b in situations]
        todo: dict[tuple, Situation] = {}
        for key, situation in zip(keys, situations):
            if key not in self._cache and key not in todo:
                todo[key] = situation
        if todo:
            pending = list(todo.items())
            batch = [s for _, s in pending]
            if hasattr(self._model, "predict_proba_many"):
                values = self._model.predict_proba_many(map_name, batch)
            else:
                values = [self._model.predict_proba(map_name, a, b) for a, b in batch]
            self.model_calls += len(pending)
            for (key, _), value in zip(pending, values):
                self._cache[key] = value
        return [self._cache[k] for k in keys]


def _shortlist(evaluator: _CachedEvaluator, map_name: Optional[str], available: list[str], k: int) -> list[str]:
    """Bounds how many candidates a greedy fill has to consider: rank by a
    context-free "how strong is this brawler on an otherwise-empty team A"
    proxy and keep only the top k. The real per-side, per-slot choice in
    _static_eval still uses the full matchup-aware model."""
    if len(available) <= k:
        return list(available)
    values = evaluator.predict_many(map_name, [([c], []) for c in available])
    ranked = sorted(zip(available, values), key=lambda cv: cv[1], reverse=True)
    return [c for c, _ in ranked[:k]]


def _static_eval(evaluator: _CachedEvaluator, state: DraftState, pool: list[str], shortlist_size: int = 12) -> float:
    """
    Evaluator used both at a true terminal state and at a search-depth
    cutoff. Plain model.predict_proba(picks_so_far) ignores bans entirely -
    banning a brawler doesn't add them to anyone's `picks`, so at a cutoff
    that hasn't reached the pick phase yet, every ban candidate would look
    identical (same empty picks either way). That would make ban
    recommendations meaningless at any depth shallow enough to matter for a
    live second-screen tool.

    Instead, fill each side's remaining pick slots with a cheap greedy
    choice from the currently available pool (which already excludes
    whatever's been banned) before evaluating. This reflects "if both sides
    just took their best remaining options from here" - exactly the
    potential a ban is trying to deny (Abschnitt 1a), and it collapses to
    the plain terminal-state evaluation once both sides already have all
    their picks (the fill loop is then a no-op).

    The fill follows the draft's REAL pick order (rules.turn_order(): who has
    first pick, the 1-2-2-1 blocks), not a plain A/B/A/B alternation. It
    matters most for the very thing bans are for: if the enemy picks first,
    our ban of the strongest brawler only has value because the enemy would
    have taken it first - an alternation that lets A grab it first
    undervalues that ban (found by the unit tests in tests/test_search.py).
    """
    slots = state.rules.picks_per_team
    a, b = list(state.picks[Team.A]), list(state.picks[Team.B])
    if len(a) >= slots and len(b) >= slots:
        return evaluator.predict_proba(state.map_name, a, b)

    remaining = _shortlist(evaluator, state.map_name, state.available_brawlers(pool), shortlist_size)

    pick_turns = [t.team for t in state.rules.turn_order() if t.phase is Phase.PICK]
    for team in pick_turns[len(a) + len(b):]:
        if not remaining:
            break
        if team is Team.A:
            values = evaluator.predict_many(state.map_name, [(a + [c], b) for c in remaining])
            a.append(remaining.pop(max(range(len(remaining)), key=values.__getitem__)))
        else:
            values = evaluator.predict_many(state.map_name, [(a, b + [c]) for c in remaining])
            b.append(remaining.pop(min(range(len(remaining)), key=values.__getitem__)))  # B wants P(A wins) low

    return evaluator.predict_proba(state.map_name, a, b)


def candidate_scores(
    evaluator: _CachedEvaluator, state: DraftState, pool: list[str], turn: Turn
) -> dict[str, float]:
    """
    Cheap, non-recursive one-ply score per available brawler, used only to
    shortlist candidates before the expensive minimax expansion below - not
    the final searched value.

    Pick turn: score = P(turn.team wins) if they picked this brawler right
    now - the direct "how good does this look immediately" heuristic used
    purely for move ordering (which also makes alpha-beta pruning much more
    effective, since it sees the best move first).

    Ban turn: score = how dangerous this brawler would be *left on the
    table for the opponent* - P(the opponent wins) if their current picks
    were joined by this candidate, against our own current picks. This
    mirrors the spec's own worked example (Abschnitt 1a): the best bans are
    the ones with the highest winrate against our own strongest available
    picks, not globally highest winrate.
    """
    available = state.available_brawlers(pool)
    a, b = state.picks[Team.A], state.picks[Team.B]

    # Both cases score "candidate joins the side X" - for a pick, X is the
    # mover; for a ban, X is the opponent (whom the ban denies).
    beneficiary = turn.team if turn.phase is Phase.PICK else turn.team.other()
    if beneficiary is Team.A:
        values = evaluator.predict_many(state.map_name, [(a + [c], b) for c in available])
    else:
        values = [1.0 - v for v in evaluator.predict_many(state.map_name, [(a, b + [c]) for c in available])]
    return dict(zip(available, values))


def top_candidates(
    evaluator: _CachedEvaluator, state: DraftState, pool: list[str], turn: Turn, n: int
) -> list[str]:
    scores = candidate_scores(evaluator, state, pool, turn)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, _ in ranked[:n]]


def minimax(
    evaluator: _CachedEvaluator,
    state: DraftState,
    pool: list[str],
    depth: int,
    alpha: float = NEG_INF,
    beta: float = POS_INF,
    top_n: int = 8,
) -> tuple[float, Optional[str]]:
    """
    Returns (value, best_brawler_for_the_current_turn). `value` is always
    P(Team A wins) under the searched lines; see module docstring for why
    Team A maximizes and Team B minimizes it on their respective turns.
    Raises _OutOfTime if the evaluator's deadline passes mid-search.
    """
    evaluator.check_time()
    turn = state.current_turn()
    if turn is None or depth == 0:
        return _static_eval(evaluator, state, pool), None

    candidates = top_candidates(evaluator, state, pool, turn, top_n)
    if not candidates:
        return _static_eval(evaluator, state, pool), None

    maximizing = turn.team is Team.A
    best_value = NEG_INF if maximizing else POS_INF
    best_choice = candidates[0]

    for cand in candidates:
        child = state.apply(cand)
        value, _ = minimax(evaluator, child, pool, depth - 1, alpha, beta, top_n)
        if maximizing:
            if value > best_value:
                best_value, best_choice = value, cand
            alpha = max(alpha, best_value)
        else:
            if value < best_value:
                best_value, best_choice = value, cand
            beta = min(beta, best_value)
        if beta <= alpha:
            break  # alpha-beta cutoff: rest of the candidates can't change the outcome

    return best_value, best_choice


@dataclass
class Recommendation:
    brawler: str
    win_probability: float  # P(the mover's own team wins) after this action, under optimal continuation
    delta: float  # vs. the mover's win probability before acting


def recommend(
    model: WinProbabilityModel,
    state: DraftState,
    pool: list[str],
    depth: int = 4,
    top_n: int = 16,
    num_alternatives: int = 3,
    inner_top_n: int = 8,
    time_budget: Optional[float] = None,
    stats: Optional[dict] = None,
) -> list[Recommendation]:
    """
    Top recommendation plus up to `num_alternatives` runners-up for whoever's
    turn it currently is (Abschnitt 9 "Ausgabe": recommended brawler,
    resulting win probability, alternatives).

    Searches iteratively deeper (1 ply, 2 plies, ... up to `depth`) and
    answers from the last completed iteration. With `time_budget` (seconds),
    an iteration that runs out of time is discarded; depth 1 always
    completes so there's always an answer. `stats`, if given, is filled with
    the depth reached, elapsed seconds and number of model evaluations.
    """
    turn = state.current_turn()
    if turn is None:
        return []

    started = time.perf_counter()
    evaluator = _CachedEvaluator(model)
    mover_is_a = turn.team is Team.A
    candidates = top_candidates(evaluator, state, pool, turn, max(top_n, num_alternatives + 1))
    baseline = _static_eval(evaluator, state, pool)
    baseline_for_mover = baseline if mover_is_a else 1.0 - baseline

    scored: list[tuple[str, float]] = []
    depth_reached = 0
    for d in range(1, max(depth, 1) + 1):
        evaluator.deadline = started + time_budget if (time_budget is not None and d > 1) else None
        try:
            iteration = [
                (cand, minimax(evaluator, state.apply(cand), pool, d - 1, top_n=inner_top_n)[0])
                for cand in candidates
            ]
        except _OutOfTime:
            break
        scored, depth_reached = iteration, d

    scored.sort(key=lambda kv: kv[1] if mover_is_a else -kv[1], reverse=True)

    if stats is not None:
        stats.update(
            depth=depth_reached,
            elapsed=time.perf_counter() - started,
            model_calls=evaluator.model_calls,
        )

    out: list[Recommendation] = []
    for brawler, value in scored[: num_alternatives + 1]:
        mover_prob = value if mover_is_a else 1.0 - value
        out.append(Recommendation(
            brawler=brawler,
            win_probability=mover_prob,
            delta=mover_prob - baseline_for_mover,
        ))
    return out
