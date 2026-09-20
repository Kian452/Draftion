"""
Feature encoding for the win-probability model (Abschnitt 8).

Design driven directly by the data-limitation note in the spec (Abschnitt 6):
with ~70+ brawlers there are ~55k possible 3-brawler compositions, so the
exact 6-brawler draft situation (3 vs 3) essentially never repeats even
across hundreds of thousands of games. One-hot-encoding the full situation
as a single atomic feature would mean the model almost never sees the same
feature twice. Instead every situation is decomposed into small, frequently
recurring building blocks:

    - per-brawler presence, kept separate per side (brawler_a:X / brawler_b:X)
    - teammate-pair interactions, per side (pair_a:X|Y / pair_b:X|Y)
    - opponent-pair (matchup) interactions, across sides (matchup:X|Y)
    - map

These are emitted as a sparse dict and vectorized with a DictVectorizer, so:
  - a brawler/pair that never appears just never gets a column (no wasted
    dense space across ~70^2 possible pairs),
  - a partial draft state (some slots not yet filled) naturally produces
    fewer features rather than needing explicit "empty slot" padding values -
    missing slots simply contribute nothing, which is exactly the "model
    must also work on incomplete teams mid-draft" requirement in Abschnitt 8.
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional, Sequence

Team = Sequence[Optional[str]]


def encode_situation(map_name: Optional[str], team_a: Team, team_b: Team) -> dict[str, float]:
    """
    team_a / team_b: up to 3 brawler names each; missing slots may be
    represented as None, "", or simply omitted - all treated as empty.

    Returns a sparse {feature_name: 1.0} dict ready for DictVectorizer.
    """
    a = [b for b in team_a if b]
    b = [b for b in team_b if b]

    feats: dict[str, float] = {"bias": 1.0}
    for x in a:
        feats[f"brawler_a:{x}"] = 1.0
    for x in b:
        feats[f"brawler_b:{x}"] = 1.0
    for x, y in combinations(sorted(a), 2):
        feats[f"pair_a:{x}|{y}"] = 1.0
    for x, y in combinations(sorted(b), 2):
        feats[f"pair_b:{x}|{y}"] = 1.0
    for x in a:
        for y in b:
            feats[f"matchup:{x}|{y}"] = 1.0
    if map_name:
        feats[f"map:{map_name}"] = 1.0
    return feats


def encode_dataframe_rows(df) -> list[dict[str, float]]:
    """Vectorized helper for training: one feature dict per training-table row."""
    return [
        encode_situation(row["map"], row["team_a"], row["team_b"])
        for _, row in df.iterrows()
    ]
