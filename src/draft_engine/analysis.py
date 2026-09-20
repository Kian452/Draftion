"""
Post-draft analysis (the results screen): why the model rates a finished draft
the way it does.

Everything is derived from the same win-probability model as the
recommendations, using *partial teams as probes* - the model supports those by
design (Abschnitt 8) - so it works for the logistic-regression and the
gradient-boosting backend alike, without peeking at coefficients. All values
are probabilities of Team A winning unless stated otherwise; team A is the
user's own team.

  win_probability_a  the full 3v3 rating
  matchups[i][j]     head-to-head chance that A's brawler i beats B's brawler j,
                     evaluated 1v1 and averaged with the mirrored evaluation
                     (so the model's arbitrary "side A" bias cancels out). It
                     contains both brawlers' general strength and their
                     counter effect.
  impact             leave-one-out: how much a brawler moves its own team's win
                     chance versus the same draft without it (positive = helps
                     its own team)
  synergy            teammate pair effect beyond the two brawlers' individual
                     strength (inclusion-exclusion on 0/1/2-brawler teams
                     against an empty opponent); positive = helps that team

The 1v1 / 2v0 probes are outside what the model was trained on (always 3v3),
so treat the numbers as the model's opinion, not measured winrates.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Optional, Protocol, Sequence

Situation = tuple[Sequence[str], Sequence[str]]


class _BatchModel(Protocol):
    def predict_proba_many(self, map_name: Optional[str], situations: Sequence[Situation]) -> list[float]: ...


class _Batch:
    """Collects the situations to score so the model is called once."""

    def __init__(self) -> None:
        self._situations: list[Situation] = []
        self._index: dict[tuple, int] = {}
        self._values: list[float] = []

    def add(self, team_a: Sequence[str], team_b: Sequence[str]) -> tuple:
        key = (tuple(sorted(team_a)), tuple(sorted(team_b)))
        if key not in self._index:
            self._index[key] = len(self._situations)
            self._situations.append((list(team_a), list(team_b)))
        return key

    def run(self, model: _BatchModel, map_name: Optional[str]) -> None:
        self._values = model.predict_proba_many(map_name, self._situations)

    def get(self, key: tuple) -> float:
        return self._values[self._index[key]]


def analyze_draft(
    model: _BatchModel, map_name: Optional[str], team_a: Sequence[str], team_b: Sequence[str]
) -> dict[str, Any]:
    a, b = list(team_a), list(team_b)
    batch = _Batch()

    full = batch.add(a, b)
    head_to_head = {(x, y): (batch.add([x], [y]), batch.add([y], [x])) for x in a for y in b}
    drop_a = {x: batch.add([m for m in a if m != x], b) for x in a}
    drop_b = {y: batch.add(a, [m for m in b if m != y]) for y in b}
    empty = batch.add([], [])
    synergy_a = {
        (x, y): (batch.add([x, y], []), batch.add([x], []), batch.add([y], []))
        for x, y in combinations(a, 2)
    }
    synergy_b = {
        (x, y): (batch.add([], [x, y]), batch.add([], [x]), batch.add([], [y]))
        for x, y in combinations(b, 2)
    }

    batch.run(model, map_name)
    p_a = batch.get(full)

    return {
        "team_a": a,
        "team_b": b,
        "win_probability_a": p_a,
        "matchups": [
            [0.5 * (batch.get(head_to_head[(x, y)][0]) + 1.0 - batch.get(head_to_head[(x, y)][1])) for y in b]
            for x in a
        ],
        "impact_a": [{"brawler": x, "value": p_a - batch.get(drop_a[x])} for x in a],
        # Removing an enemy brawler *raises* A's chance if that brawler was
        # helping B, so its impact for B is the increase of P(A wins).
        "impact_b": [{"brawler": y, "value": batch.get(drop_b[y]) - p_a} for y in b],
        "synergy_a": [
            {"pair": [x, y], "value": batch.get(both) - batch.get(one) - batch.get(two) + batch.get(empty)}
            for (x, y), (both, one, two) in synergy_a.items()
        ],
        "synergy_b": [
            {"pair": [x, y], "value": -(batch.get(both) - batch.get(one) - batch.get(two) + batch.get(empty))}
            for (x, y), (both, one, two) in synergy_b.items()
        ],
    }
