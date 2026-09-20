"""
Once you've crawled some data, run `python -m src.extract types` first to
see which (type, mode) combination corresponds to Ranked in your dataset -
Supercell has renamed/reshuffled mode names before, so verify against real
data instead of assuming a string.

What real data showed (2026-09-19): Ranked draft battles are type
`soloRanked` (only the six 3v3 modes gemGrab/brawlBall/knockout/hotZone/heist/
bounty). Type `ranked` is ordinary trophy play in every mode incl. Showdown -
NOT the draft. And `battle.rank` doesn't exist for soloRanked; each player's
`brawler.trophies` holds their Ranked tier there (4..18 seen), so the battle's
`rank` column is the mean over all six players (see db.battle_rank).

Then run `python -m src.extract ranks soloRanked` to see the distribution of
that value. The exact number-to-tier mapping (which number means "Legendary
III", which means "Pro") isn't documented, so don't assume it - cross-check
against a known rank of your own (your own soloRanked battles show your
brawler's value). Once you know the right cutoff, pass it as `min_rank` below.

build_training_table() then turns raw battles into a flat, weighted table:
(map, team_a_brawlers[3], team_b_brawlers[3], winner, weight) - the input
format the model (next step, once you have enough real data) trains on.
"""
from __future__ import annotations

import sys

import pandas as pd

from .db import connect, distinct_battle_types


def inspect_types(db_path: str = "data/brawl.db") -> None:
    conn = connect(db_path)
    rows = distinct_battle_types(conn)
    if not rows:
        print("No battles in the DB yet - run the crawler first.")
        return
    print(f"{'type':<15}{'mode':<20}count")
    for type_, mode, count in rows:
        print(f"{str(type_):<15}{str(mode):<20}{count}")


def inspect_ranks(db_path: str = "data/brawl.db", battle_type: str = "soloRanked") -> None:
    conn = connect(db_path)
    rows = conn.execute(
        "SELECT rank, COUNT(*) FROM battles WHERE type = ? GROUP BY rank ORDER BY rank",
        (battle_type,),
    ).fetchall()
    if not rows:
        print(f"No battles with type='{battle_type}' found - run inspect_types() first "
              f"to get the right type string.")
        return
    print(f"{'rank':<8}count")
    for rank, count in rows:
        print(f"{str(rank):<8}{count}")


def inspect_status(db_path: str = "data/brawl.db") -> None:
    """How far the data collection is: battles that count for training (Ranked type, lobby rank
    >= MIN_RANK, winner known), how fast they arrive, and how many players were polled."""
    from datetime import datetime, timedelta

    from .config import Settings

    s = Settings.from_env()
    conn = connect(db_path)
    one = lambda sql, *args: conn.execute(sql, args).fetchone()[0]
    total = one("SELECT COUNT(*) FROM battles WHERE type = ?", s.ranked_type)
    usable = one("SELECT COUNT(*) FROM battles WHERE type = ? AND rank >= ? AND winner_team IS NOT NULL",
                 s.ranked_type, s.min_rank)
    newest = one("SELECT MAX(battle_time) FROM battles WHERE type = ?", s.ranked_type)
    print(f"{s.ranked_type} battles stored:        {total:>8,}")
    print(f"  usable for training (rank >= {s.min_rank}):  {usable:>6,}   <- this is the number that counts")
    if newest:
        cutoff = (datetime.strptime(newest[:15], "%Y%m%dT%H%M%S") - timedelta(days=1)).strftime("%Y%m%dT%H%M%S")
        day = one("SELECT COUNT(*) FROM battles WHERE type = ? AND rank >= ? AND battle_time >= ?",
                  s.ranked_type, s.min_rank, cutoff)
        print(f"  usable, played in the 24 h before the newest battle: {day:,}")
    print(f"players polled at least once:    {one('SELECT COUNT(*) FROM players WHERE last_crawled IS NOT NULL'):>8,}")
    last = one("SELECT MAX(last_crawled) FROM players")
    print(f"last poll:                       {last or 'never'}")
    print(f"\nA trained model is only built from {s.min_train_rows:,} usable battles and only takes over from the "
          "raw win rates once it beats them in cross-validation.\n"
          "Rough goal for a useful brawler-level model: 40,000+ usable battles "
          "(python -m src.tune shows how the numbers look on the current data).")


def rank_weight(rank: int, min_rank: int, weight_step: float = 1.0) -> float:
    """Linear weight: the higher above min_rank, the more it counts.
    Tune weight_step once you've seen real rank counts (e.g. lower it if
    the top ranks are so rare that they'd dominate the loss)."""
    return 1.0 + (rank - min_rank) * weight_step


def build_training_table(db_path: str, battle_type: str, min_rank: int, weight_step: float = 1.0) -> pd.DataFrame:
    """
    battle_type: the value confirmed via inspect_types() as Ranked.
    min_rank: the numeric rank confirmed via inspect_ranks() as Legendary III
              (or wherever you want the cutoff) - only battles at or above
              this are included, weighted higher the further above it.
    """
    conn = connect(db_path)
    battles = pd.read_sql_query(
        """SELECT battle_hash, battle_time, map, winner_team, rank FROM battles
           WHERE type = ? AND rank >= ? AND winner_team IS NOT NULL""",
        conn, params=(battle_type, min_rank),
    )
    # Only the players of the selected battles, and grouped in one pass: looking
    # each battle's players up with a filter over the whole table is quadratic
    # and takes hours once a few tens of thousands of battles are collected.
    bp = pd.read_sql_query(
        """SELECT bp.battle_hash, bp.team, bp.brawler_name FROM battle_players bp
           JOIN battles b ON b.battle_hash = bp.battle_hash
           WHERE b.type = ? AND b.rank >= ? AND b.winner_team IS NOT NULL
           ORDER BY bp.rowid""",
        conn, params=(battle_type, min_rank),
    )
    teams_of: dict[str, dict[int, list[str]]] = {}
    for bhash, team, name in zip(bp["battle_hash"], bp["team"], bp["brawler_name"]):
        teams_of.setdefault(bhash, {}).setdefault(team, []).append(name)

    rows = []
    for b in battles.itertuples(index=False):
        teams = teams_of.get(b.battle_hash, {})
        # winner_team indexes into the raw battle.teams array (0 or 1) - the
        # same indexing used for the `team` column in battle_players, so it
        # lines up directly with team_a/team_b below regardless of which
        # physical team happened to be listed first in the API response.
        if len(teams) != 2 or any(len(t) != 3 for t in teams.values()) or b.winner_team not in teams:
            continue  # skip incomplete data / non-3v3 modes
        first, second = sorted(teams)
        rows.append({
            "battle_time": b.battle_time,
            "map": b.map,
            "team_a": teams[first],
            "team_b": teams[second],
            "winner": 1 if first == b.winner_team else 0,
            "rank": b.rank,
            "weight": rank_weight(b.rank, min_rank, weight_step),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "types"
    if cmd == "ranks":
        # optional 3rd argument: the type value `types` showed for Ranked, e.g. `ranks soloRanked`
        inspect_ranks(battle_type=sys.argv[2] if len(sys.argv) > 2 else "soloRanked")
    elif cmd == "status":
        inspect_status()
    else:
        inspect_types()
