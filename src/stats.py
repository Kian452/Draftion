"""
Exploratory brawler statistics computed from the flat training table
(build_training_table() in extract.py).

Careful: with ~70+ brawlers, most specific combinations (brawler A vs B,
with C as teammate, on map M) are seen far too rarely to trust a raw win
rate - a 65% winrate over 8 games is noise, not signal. These functions
only compute *aggregate* one-brawler and one-pair stats (which have far
more samples per cell) and drop anything below `min_samples`.

Also: a raw "win rate when paired with X" conflates two different things -
genuine synergy, and X just being a strong brawler that lifts any team it's
on. That confound is exactly what the trained model (logistic regression /
gradient boosting on brawler indicators, planned next) is built to
disentangle - it's the ML equivalent of "adjusted plus-minus" in sports
analytics. Treat everything here as exploration/sanity-checking, not as the
final counter/synergy source of truth for the draft engine.
"""
from __future__ import annotations

from itertools import combinations

import pandas as pd


def _weighted_winrate(g: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "games": len(g),
        "winrate": (g["win"] * g["weight"]).sum() / g["weight"].sum(),
    })


def brawler_winrate(df: pd.DataFrame, min_samples: int = 30) -> pd.DataFrame:
    """Overall win rate per brawler, regardless of teammates/opponents."""
    rows = []
    for _, r in df.iterrows():
        for b in r["team_a"]:
            rows.append({"brawler": b, "win": r["winner"], "weight": r["weight"]})
        for b in r["team_b"]:
            rows.append({"brawler": b, "win": 1 - r["winner"], "weight": r["weight"]})
    long = pd.DataFrame(rows)
    agg = long.groupby("brawler").apply(_weighted_winrate).reset_index()
    return agg[agg["games"] >= min_samples].sort_values("winrate", ascending=False)


def synergy_matrix(df: pd.DataFrame, min_samples: int = 30) -> pd.DataFrame:
    """Win rate for every teammate pair that appeared together (same team)."""
    rows = []
    for _, r in df.iterrows():
        for team, win in ((r["team_a"], r["winner"]), (r["team_b"], 1 - r["winner"])):
            for a, b in combinations(sorted(team), 2):
                rows.append({"pair": f"{a} + {b}", "win": win, "weight": r["weight"]})
    long = pd.DataFrame(rows)
    agg = long.groupby("pair").apply(_weighted_winrate).reset_index()
    return agg[agg["games"] >= min_samples].sort_values("winrate", ascending=False)


def matchup_matrix(df: pd.DataFrame, min_samples: int = 30) -> pd.DataFrame:
    """
    Rough counter indicator: among matches where a team with brawler A faced
    a team containing brawler B, how often did A's team win. This mixes real
    counter interaction with each brawler's general strength (see module
    docstring) - use as a first pass, not ground truth.
    """
    rows = []
    for _, r in df.iterrows():
        for a in r["team_a"]:
            for b in r["team_b"]:
                rows.append({"a": a, "b": b, "win": r["winner"], "weight": r["weight"]})
        for a in r["team_b"]:
            for b in r["team_a"]:
                rows.append({"a": a, "b": b, "win": 1 - r["winner"], "weight": r["weight"]})
    long = pd.DataFrame(rows)
    agg = long.groupby(["a", "b"]).apply(_weighted_winrate).reset_index()
    return agg[agg["games"] >= min_samples].sort_values("winrate", ascending=False)
