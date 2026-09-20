"""
Fallback win-probability model built from raw brawler win rates, for the time
before there is enough data to train the real model on.

Without it the app has two bad options early on: no recommendations at all, or
a trained model that has seen only a few hundred battles and mostly memorizes
noise. This one is deliberately simple and hard to overfit: every brawler's win
rate (stats.brawler_winrate) is shrunk towards 50% by `prior_games` imaginary
games, turned into a strength on the log-odds scale, and a team's strength is
the sum of its brawlers'. So

    P(A wins) = sigmoid(scale * (sum(strength of A) - sum(strength of B)))

It has no notion of counters, synergies or maps - that is exactly what the
trained model adds once it has the data (see stats.py for why raw counts can't
separate a real synergy from "X is just a strong brawler"). The app therefore
labels everything based on this model as preliminary / low confidence.

The two knobs (`prior_games`, `scale`) were measured with `python -m src.tune`
(time-series CV, ~1.5k real battles): the raw win rates carry a real but small
signal (AUC ~0.57, log loss 0.685 vs 0.693 for always saying 50%), and how hard
they are shrunk decides whether that shows or the model is just overconfident -
prior 20 / scale 1.0 (the naive choice) scores 0.698, WORSE than a coin flip.
prior 100 / scale 0.5 is within one standard error of the best setting while
keeping the win chance of real drafts within about 40-60% (the most lopsided real
draft in the cross-validation reached 60%; the theoretical extreme - the three best
brawlers against the three worst - is about 70%), which is the honest range here.

Same interface as model.WinProbabilityModel (predict_proba_many, partial teams
allowed), so the draft search doesn't care which one it is talking to.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import pandas as pd

from .stats import brawler_winrate

KIND = "raw-stats"


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


@dataclass
class RawStatsModel:
    strength: dict[str, float]  # brawler -> log-odds strength (0 = average)
    n_battles: int = 0
    backend: str = KIND
    data_source: str = "real"
    scale: float = 0.5
    brawlers: list[str] = field(default_factory=list)

    kind = KIND

    @staticmethod
    def from_table(
        df: pd.DataFrame,
        prior_games: float = 100.0,
        scale: float = 0.5,
        brawlers: Optional[list[str]] = None,
        data_source: str = "real",
    ) -> "RawStatsModel":
        """`df` is extract.build_training_table()'s output."""
        strength: dict[str, float] = {}
        if len(df):
            # min_samples=1: the shrinkage below does the noise control, so a
            # brawler seen twice isn't dropped, just pulled hard towards 50%.
            for row in brawler_winrate(df, min_samples=1).itertuples(index=False):
                shrunk = (row.winrate * row.games + 0.5 * prior_games) / (row.games + prior_games)
                strength[row.brawler] = _logit(shrunk)
        return RawStatsModel(
            strength=strength, n_battles=len(df), scale=scale,
            brawlers=brawlers or [], data_source=data_source,
        )

    def known_brawlers(self) -> set[str]:
        return set(self.strength)

    def predict_proba_many(
        self,
        map_name: Optional[str],  # unused: too few battles per map to say anything
        situations: Sequence[tuple[Sequence[Optional[str]], Sequence[Optional[str]]]],
    ) -> list[float]:
        out = []
        for team_a, team_b in situations:
            z = sum(self.strength.get(b, 0.0) for b in team_a if b) - sum(
                self.strength.get(b, 0.0) for b in team_b if b
            )
            out.append(1.0 / (1.0 + math.exp(-self.scale * z)))
        return out

    def predict_proba(self, map_name, team_a, team_b) -> float:
        return self.predict_proba_many(map_name, [(team_a, team_b)])[0]
