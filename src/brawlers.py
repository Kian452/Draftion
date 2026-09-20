"""
Brawler roster used by the model's feature space, the draft engine's
candidate pool, and the frontend's pick/ban grid (with portraits).

The full current roster (100+ brawlers, and Supercell adds more regularly)
is not something to hardcode by memory - a wrong/incomplete list would
silently corrupt the model's feature space (a brawler missing here can never
be picked/banned in the UI or trained on) and the draft search (a stale extra
name could be searched even though it doesn't exist). So this module treats
live APIs as the only source of truth:

    python -m src.brawlers fetch

writes data/brawlers.json as a list of {"id", "name", "image"} objects:

  - id + name come from the official Brawl Stars API when BS_API_TOKEN is
    set (names there are UPPERCASE, e.g. "EL PRIMO" - exactly the strings
    the battlelogs, and therefore the trained model, use).
  - image comes from Brawlify (community CDN, no key needed), joined by the
    numeric brawler id.
  - Without a token, the roster falls back to Brawlify's own list with names
    upper-cased to match the official spelling. That works for the UI right
    away, but is unverified against real battlelog data - run fetch again
    once you have a token.

Everything else in the project loads through load_brawlers() /
load_brawler_details(). Until `fetch` has been run once, they fall back to
SEED_BRAWLERS - a short, high-confidence list of long-standing brawlers,
enough to smoke-test the pipeline end-to-end, NOT a claim of completeness.
Don't extend SEED_BRAWLERS by memory; run `fetch` instead.
"""
from __future__ import annotations

import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger("brawlers")

DEFAULT_PATH = "data/brawlers.json"
BRAWLIFY_URL = "https://api.brawlapi.com/v1/brawlers"

SEED_BRAWLERS: list[str] = [
    "SHELLY", "COLT", "BULL", "BROCK", "RICO", "SPIKE", "BARLEY", "JESSIE",
    "NITA", "DYNAMIKE", "EL PRIMO", "MORTIS", "CROW", "POCO", "BO", "PIPER",
    "PAM", "FRANK", "BIBI", "BEA", "NANI", "DARRYL", "PENNY", "CARL",
]


def load_brawler_details(path: str = DEFAULT_PATH) -> list[dict[str, Any]]:
    """[{"name": str, "id": int | None, "image": str | None}, ...]"""
    p = Path(path)
    if p.exists():
        details = json.loads(p.read_text(encoding="utf-8"))
        if details:
            return details
    log.warning(
        "%s not found or empty - falling back to a %d-brawler seed list without "
        "portraits. Run `python -m src.brawlers fetch` to load the real roster.",
        path, len(SEED_BRAWLERS),
    )
    return [{"name": n} for n in SEED_BRAWLERS]


def load_brawlers(path: str = DEFAULT_PATH) -> list[str]:
    return [b["name"] for b in load_brawler_details(path)]


def _fetch_brawlify() -> list[dict[str, Any]]:
    resp = requests.get(BRAWLIFY_URL, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return data.get("list", []) if isinstance(data, dict) else data


def _image_exists(url: str | None) -> bool:
    if not url:
        return False
    try:
        return requests.head(url, timeout=15, allow_redirects=True).status_code == 200
    except requests.RequestException:
        return False


def drop_dead_images(entries: list[dict[str, Any]]) -> None:
    """Brawlify lists portrait URLs for brand-new brawlers before the image
    files are actually uploaded (they 404). Checking here means the roster file
    only ever holds working links, instead of every page load hitting a 404."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        alive = list(pool.map(_image_exists, [e["image"] for e in entries]))
    for entry, ok in zip(entries, alive):
        if not ok:
            entry["image"] = None


def fetch_and_save(path: str = DEFAULT_PATH) -> list[dict[str, Any]]:
    from .api_client import BrawlStarsAPIError, BrawlStarsClient  # not needed unless fetching

    try:
        brawlify = _fetch_brawlify()
    except requests.RequestException as e:
        log.warning("Could not reach Brawlify (%s) - continuing without portraits.", e)
        brawlify = []
    # "imageUrl2" is the borderless portrait - it sits cleanly inside our own
    # rounded tiles instead of clashing with the rarity-colored frame.
    images = {b["id"]: b.get("imageUrl2") for b in brawlify}

    official: list[dict[str, Any]] = []
    try:
        official = BrawlStarsClient().get_brawlers()
    except RuntimeError:
        log.warning("No BS_API_TOKEN set - can't use the official API for the roster.")
    except BrawlStarsAPIError as e:
        log.warning("Official API request failed (%s) - can't use it for the roster.", e)

    if official:
        entries = [{"id": b["id"], "name": b["name"], "image": images.get(b["id"])} for b in official]
    elif brawlify:
        log.warning(
            "Using Brawlify's roster with names upper-cased to mimic the official "
            "spelling. Unverified against real battlelog data - re-run with "
            "BS_API_TOKEN set to get the official names."
        )
        entries = [{"id": b["id"], "name": b["name"].upper(), "image": b.get("imageUrl2")} for b in brawlify]
    else:
        raise RuntimeError("Neither the official API nor Brawlify could be reached - no roster to save.")

    entries.sort(key=lambda e: e["name"])
    drop_dead_images(entries)
    missing = [e["name"] for e in entries if not e["image"]]
    if missing:
        log.warning(
            "No working portrait for %d brawlers (the UI shows initials instead; "
            "re-run fetch later, new brawlers' images usually appear within days): %s",
            len(missing), missing,
        )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %d brawlers to %s", len(entries), path)
    return entries


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    if cmd == "fetch":
        fetch_and_save()
    else:
        print(f"Unknown command: {cmd}. Usage: python -m src.brawlers fetch")
