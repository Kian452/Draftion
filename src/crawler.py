"""
Snowball crawler: starts from top global (and optionally per-country)
players by trophy ranking, then follows every opponent/teammate tag found
in their battlelogs.

There's no dedicated Ranked-tier leaderboard in the official API - only a
trophy (Trophy Road) ranking, which is a correlated but imperfect proxy for
Ranked skill. (Checked live on 2026-09-19: GET /rankings/{global,US,CH}/powerplay/
seasons, .../seasons/latest and .../seasons/<id> all answer 404 while the ordinary
/rankings/global/players answers 200 with the same key - the Power Play leaderboard
is gone, so there is no better seed source. Don't re-test it.)
To compensate, the frontier is a priority queue: a
newly-discovered tag's priority is the `rank` of the battle it was found
in, so players seen in high-rank Ranked matches get crawled before players
only seen in casual/low-rank games. This steers the snowball into the
high-rank subgraph instead of diluting equally across all of it.

High-rank players form a small, tightly-connected pool (matchmaking keeps
pairing the same crowd), so a snowball run saturates it fast and then runs
out of new tags. Use --reseed-from-rank on later runs to also re-enqueue
every already-known player at or above that rank, so the crawler goes back
and polls the same pool for whatever new matches they've played since -
battlelog only ever shows the last ~25, so this is an ongoing collection
job, not a one-time backlog to clear.

Since the API only exposes each player's ~25 most recent battles (with up to
a 30 min delay for new ones) and there's no bulk endpoint, you need to
re-run this repeatedly over days/weeks (e.g. via cron) to accumulate enough
Ranked matches for training.

Usage:
    # first run: broad discovery from trophy rankings
    python -m src.crawler --max-players 2000

    # later runs: keep polling the known high-rank pool for new matches
    python -m src.crawler --max-players 2000 --reseed-from-rank 18 --recrawl-hours 2
"""
from __future__ import annotations

import argparse
import heapq
import itertools
import logging
import time

from .api_client import BrawlStarsAPIError, BrawlStarsClient
from .db import (
    battle_rank,
    connect,
    has_player_been_crawled_recently,
    high_rank_player_tags,
    mark_player_crawled,
    store_battle,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("crawler")


def seed_tags(client: BrawlStarsClient, countries: list[str], limit_per_country: int) -> list[str]:
    tags: list[str] = []
    for country in countries:
        try:
            ranking = client.get_rankings("players", country=country, limit=limit_per_country)
            tags.extend(p["tag"] for p in ranking)
        except BrawlStarsAPIError as e:
            log.warning("Could not fetch rankings for %s: %s", country, e)
    return list(dict.fromkeys(tags))  # dedupe, keep order


def crawl(
    db_path: str,
    countries: list[str],
    seed_limit: int,
    max_players: int,
    sleep_between_players: float = 0.2,
    recrawl_hours: int = 6,
    reseed_from_rank: int | None = None,
    reseed_battle_type: str = "soloRanked",
) -> None:
    client = BrawlStarsClient()
    conn = connect(db_path)

    # Priority queue of (-priority, tie_breaker, tag). Higher rank = crawled
    # sooner. Tie-breaker keeps tags out of comparisons (they're not orderable).
    counter = itertools.count()
    frontier: list[tuple[int, int, str]] = [
        (0, next(counter), tag) for tag in seed_tags(client, countries, seed_limit)
    ]

    if reseed_from_rank is not None:
        known = high_rank_player_tags(conn, reseed_from_rank, reseed_battle_type)
        # Big priority boost: known-good players go before freshly
        # discovered, unconfirmed ones - we're here to poll them, not explore.
        boost = reseed_from_rank + 1000
        frontier.extend((-boost, next(counter), tag) for tag in known)
        log.info("Reseeding with %d known players at rank >= %d", len(known), reseed_from_rank)

    heapq.heapify(frontier)
    visited: set[str] = set()

    new_battles = 0
    processed = 0

    while frontier and processed < max_players:
        neg_priority, _, tag = heapq.heappop(frontier)
        if tag in visited or has_player_been_crawled_recently(conn, tag, within_hours=recrawl_hours):
            continue
        visited.add(tag)

        try:
            battlelog = client.get_battlelog(tag)
        except BrawlStarsAPIError as e:
            log.warning("Skipping %s: %s", tag, e)
            continue

        mark_player_crawled(conn, tag)
        processed += 1

        for entry in battlelog:
            if store_battle(conn, entry, queried_tag=tag):
                new_battles += 1
            battle = entry.get("battle", {})
            priority = battle_rank(entry) or 0
            for team in battle.get("teams", []) or []:
                for p in team:
                    other = p.get("tag")
                    if other and other not in visited:
                        heapq.heappush(frontier, (-priority, next(counter), other))

        if processed % 25 == 0:
            log.info(
                "Processed %d players, %d new battles, frontier=%d, current priority=%d",
                processed, new_battles, len(frontier), -neg_priority,
            )

        time.sleep(sleep_between_players)

    log.info("Done. Processed %d players, stored %d new battles.", processed, new_battles)


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl Brawl Stars battles via the official API.")
    parser.add_argument("--db", default="data/brawl.db")
    parser.add_argument("--countries", nargs="+", default=["global"])
    parser.add_argument("--seed-limit", type=int, default=200)
    parser.add_argument("--max-players", type=int, default=2000)
    parser.add_argument("--recrawl-hours", type=int, default=6,
                         help="Don't re-crawl a player seen within this many hours.")
    parser.add_argument("--reseed-from-rank", type=int, default=None,
                         help="Also re-enqueue known players at/above this rank (maintenance polling).")
    parser.add_argument("--reseed-battle-type", default="soloRanked",
                         help="The `type` value confirmed via `python -m src.extract types` as Ranked.")
    args = parser.parse_args()
    crawl(
        args.db, args.countries, args.seed_limit, args.max_players,
        recrawl_hours=args.recrawl_hours,
        reseed_from_rank=args.reseed_from_rank,
        reseed_battle_type=args.reseed_battle_type,
    )


if __name__ == "__main__":
    main()
