"""SQLite storage for crawled battles."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    tag TEXT PRIMARY KEY,
    discovered_from TEXT,
    last_crawled TEXT
);

CREATE TABLE IF NOT EXISTS battles (
    battle_hash TEXT PRIMARY KEY,
    battle_time TEXT,
    mode TEXT,
    map TEXT,
    type TEXT,
    result TEXT,
    duration INTEGER,
    rank INTEGER,
    winner_team INTEGER,
    raw_json TEXT
);

CREATE TABLE IF NOT EXISTS battle_players (
    battle_hash TEXT,
    team INTEGER,
    tag TEXT,
    name TEXT,
    brawler_id INTEGER,
    brawler_name TEXT,
    power INTEGER,
    FOREIGN KEY (battle_hash) REFERENCES battles(battle_hash)
);

CREATE INDEX IF NOT EXISTS idx_battle_players_hash ON battle_players(battle_hash);
CREATE INDEX IF NOT EXISTS idx_battle_players_tag ON battle_players(tag);
"""


SCHEMA_VERSION = 1  # 1 = `rank` derived from brawler.trophies for soloRanked (see battle_rank)

# Battle types whose battlelog entries carry no `battle.rank`, but where each
# player's `brawler.trophies` holds their Ranked tier instead. Measured on real
# data (2026-09-19): in `soloRanked` battles it is 4..18, in ordinary trophy
# battles (type "ranked") it is a real trophy count (43, 118, ...) and
# `battle.rank` is the Showdown placement - so this must NOT be applied there.
RANK_FROM_BRAWLER_TROPHIES = ("soloRanked",)


def connect(db_path: str = "data/brawl.db") -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    # WAL lets the web app read while the background crawler writes (they can share
    # one process on a hosted deployment); it's stored in the DB file, so this is a no-op afterwards.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    # Migrations for DBs created before these columns existed.
    for ddl in (
        "ALTER TABLE battles ADD COLUMN rank INTEGER",
        "ALTER TABLE battles ADD COLUMN winner_team INTEGER",
    ):
        try:
            conn.execute(ddl)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
    if conn.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
        backfill_ranks(conn)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
    return conn


def battle_rank(entry: dict[str, Any]) -> float | None:
    """
    The Ranked skill level of a battle (None for anything that is not a Ranked draft), for
    filtering, weighting and crawl priority.

    The spec assumed `battle.rank` (the queried player's rank). On the real API
    that field is only the Showdown placement, and it is absent from Ranked
    (soloRanked) battles altogether. There, every player's brawler carries the
    Ranked tier in `brawler.trophies`, so the mean over all six players gives a
    lobby-level rank - which needs no "keep the higher of two sightings"
    guesswork, since every participant is listed.
    """
    battle = entry.get("battle", {})
    # Deliberately not passing `battle.rank` through for other types: there it is
    # a Showdown placement (1..10), not a skill level, and would poison the
    # crawl priority and the rank filter.
    if battle.get("type") in RANK_FROM_BRAWLER_TROPHIES:
        values = [
            p.get("brawler", {}).get("trophies")
            for team in battle.get("teams") or []
            for p in team
        ]
        values = [v for v in values if isinstance(v, (int, float))]
        if values:
            return round(sum(values) / len(values), 2)
    return None


def backfill_ranks(conn: sqlite3.Connection) -> int:
    """Fill in `rank` for stored battles that predate battle_rank() (uses raw_json). Idempotent."""
    marks = ",".join("?" for _ in RANK_FROM_BRAWLER_TROPHIES)
    rows = conn.execute(
        f"SELECT battle_hash, raw_json FROM battles WHERE rank IS NULL AND type IN ({marks})",
        RANK_FROM_BRAWLER_TROPHIES,
    ).fetchall()
    updated = 0
    for bhash, raw in rows:
        rank = battle_rank(json.loads(raw))
        if rank is not None:
            conn.execute("UPDATE battles SET rank = ? WHERE battle_hash = ?", (rank, bhash))
            updated += 1
    conn.commit()
    return updated


def battle_hash(entry: dict[str, Any]) -> str:
    """
    The API gives no battle ID. The same battle shows up identically in every
    participant's battlelog, so we hash battleTime + the sorted tags of all
    players involved to dedupe across crawls.
    """
    tags: list[str] = []
    for team in entry.get("battle", {}).get("teams", []) or []:
        for p in team:
            tags.append(p.get("tag", ""))
    key = entry.get("battleTime", "") + "|" + "|".join(sorted(tags))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def mark_player_crawled(conn: sqlite3.Connection, tag: str, discovered_from: str = "") -> None:
    conn.execute(
        """INSERT INTO players (tag, discovered_from, last_crawled)
           VALUES (?, ?, datetime('now'))
           ON CONFLICT(tag) DO UPDATE SET last_crawled = excluded.last_crawled""",
        (tag, discovered_from),
    )
    conn.commit()


def has_player_been_crawled_recently(conn: sqlite3.Connection, tag: str, within_hours: int = 6) -> bool:
    row = conn.execute(
        "SELECT last_crawled FROM players WHERE tag = ? AND last_crawled >= datetime('now', ?)",
        (tag, f"-{within_hours} hours"),
    ).fetchone()
    return row is not None


def _normalize_tag(tag: str | None) -> str | None:
    if not tag:
        return None
    tag = tag.strip().upper()
    return tag if tag.startswith("#") else "#" + tag


def _resolve_winner_team(entry: dict[str, Any], queried_tag: str | None) -> int | None:
    """
    Which team (0 or 1, indexing into battle.teams) won this battle.

    The API's `battle.result` ("victory"/"defeat"/"draw") is relative to
    whichever player's battlelog was fetched - it is NOT "team index 0 won".
    `battle.teams` order isn't guaranteed to put that player's team first.
    So without knowing which team the queried player was actually on, we
    cannot safely say which team index the result refers to. If we can't
    resolve it (no queried_tag passed, tag not found in either team, or a
    draw), we return None and the battle is excluded from model training
    rather than risk a mislabeled (coin-flip) sample.
    """
    result = entry.get("battle", {}).get("result")
    if result not in ("victory", "defeat"):
        return None
    q = _normalize_tag(queried_tag)
    if q is None:
        return None
    teams = entry.get("battle", {}).get("teams", []) or []
    queried_team_idx = None
    for idx, team in enumerate(teams):
        if any(_normalize_tag(p.get("tag")) == q for p in team):
            queried_team_idx = idx
            break
    if queried_team_idx is None:
        return None
    if result == "victory":
        return queried_team_idx
    other_idx = 1 - queried_team_idx
    return other_idx if 0 <= other_idx < len(teams) else None


def store_battle(conn: sqlite3.Connection, entry: dict[str, Any], queried_tag: str | None = None) -> bool:
    """
    Returns True if this battle was newly inserted, False if already known.

    The same battle can be seen from multiple crawled players' battlelogs.
    Every sighting lists all six players, so the lobby rank (battle_rank) is
    the same each time; on a duplicate we keep the higher value anyway, in case
    a later sighting has more tier data, and fill in `winner_team`
    if we hadn't been able to resolve it yet (see _resolve_winner_team).

    `queried_tag` should be the tag of the player whose battlelog this entry
    came from - pass it so the winner can be attributed to the right team
    index instead of left as unknown.
    """
    bhash = battle_hash(entry)
    battle = entry.get("battle", {})
    new_rank = battle_rank(entry)
    new_winner_team = _resolve_winner_team(entry, queried_tag)

    existing = conn.execute(
        "SELECT rank, winner_team FROM battles WHERE battle_hash = ?", (bhash,)
    ).fetchone()
    if existing:
        old_rank, old_winner_team = existing
        updates, params = [], []
        if new_rank is not None and (old_rank is None or new_rank > old_rank):
            updates.append("rank = ?")
            params.append(new_rank)
        if old_winner_team is None and new_winner_team is not None:
            updates.append("winner_team = ?")
            params.append(new_winner_team)
        if updates:
            params.append(bhash)
            conn.execute(f"UPDATE battles SET {', '.join(updates)} WHERE battle_hash = ?", params)
            conn.commit()
        return False

    conn.execute(
        """INSERT INTO battles (battle_hash, battle_time, mode, map, type, result, duration, rank, winner_team, raw_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            bhash,
            entry.get("battleTime"),
            battle.get("mode"),
            entry.get("event", {}).get("map"),
            battle.get("type"),
            battle.get("result"),
            battle.get("duration"),
            new_rank,
            new_winner_team,
            json.dumps(entry),
        ),
    )

    for team_idx, team in enumerate(battle.get("teams", []) or []):
        for p in team:
            brawler = p.get("brawler", {})
            conn.execute(
                """INSERT INTO battle_players (battle_hash, team, tag, name, brawler_id, brawler_name, power)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    bhash,
                    team_idx,
                    p.get("tag"),
                    p.get("name"),
                    brawler.get("id"),
                    brawler.get("name"),
                    brawler.get("power"),
                ),
            )
    conn.commit()
    return True


def distinct_battle_types(conn: sqlite3.Connection) -> list[tuple[str, str, int]]:
    """Helper to figure out which (type, mode) combo corresponds to Ranked."""
    return conn.execute(
        "SELECT type, mode, COUNT(*) FROM battles GROUP BY type, mode ORDER BY 3 DESC"
    ).fetchall()


def distinct_maps(conn: sqlite3.Connection, battle_type: str) -> list[tuple[str, str | None]]:
    """(map name, mode) pairs actually seen in crawled Ranked battles, most common first.
    Mode is the battlelog's camelCase spelling, e.g. "gemGrab"."""
    rows = conn.execute(
        "SELECT map, mode, COUNT(*) c FROM battles WHERE type = ? AND map IS NOT NULL "
        "GROUP BY map, mode ORDER BY c DESC",
        (battle_type,),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def active_ranked_maps(
    conn: sqlite3.Connection, battle_type: str, window_days: int = 7, min_battles: int = 3
) -> dict[str, int]:
    """{map name: battles} for the maps in the CURRENT Ranked pool: those played at
    least `min_battles` times in the `window_days` up to the newest crawled Ranked
    battle. Measured against the live API, GET /events/rotation is not this pool -
    it lists the trophy-mode rotation, and only 5 of the 26 maps that were actually
    being played in Ranked appeared in it (Hot Zone, a Ranked mode, wasn't listed at
    all). The battles themselves are the ground truth for what is being played.
    Measured relative to the newest battle, not to today, so a database that hasn't
    been crawled for a while still yields the last known pool."""
    latest = conn.execute("SELECT MAX(battle_time) FROM battles WHERE type = ?", (battle_type,)).fetchone()[0]
    if not latest:
        return {}
    cutoff = (datetime.strptime(latest[:15], "%Y%m%dT%H%M%S") - timedelta(days=window_days)).strftime("%Y%m%dT%H%M%S")
    rows = conn.execute(
        "SELECT map, COUNT(*) FROM battles WHERE type = ? AND map IS NOT NULL AND battle_time >= ? "
        "GROUP BY map HAVING COUNT(*) >= ?",
        (battle_type, cutoff, min_battles),
    ).fetchall()
    return {name: count for name, count in rows}


def high_rank_player_tags(conn: sqlite3.Connection, min_rank: int, battle_type: str) -> list[str]:
    """
    Tags of players already seen in a Ranked battle at or above min_rank.

    High-rank players form a small, tightly-connected pool (matchmaking keeps
    throwing them together), so a snowball crawl saturates it fast and then
    has nothing new to discover. Use this to re-seed future crawl runs so
    they keep polling that same known pool for new matches instead of
    stalling once it's fully mapped - battlelog only ever shows the last
    ~25, so "new matches by known players" is an ongoing stream, not a
    one-time backlog.
    """
    rows = conn.execute(
        """SELECT DISTINCT bp.tag
           FROM battle_players bp
           JOIN battles b ON b.battle_hash = bp.battle_hash
           WHERE b.type = ? AND b.rank >= ?""",
        (battle_type, min_rank),
    ).fetchall()
    return [r[0] for r in rows]
