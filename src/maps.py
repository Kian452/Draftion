"""
Map list for the frontend's map picker.

The official Brawl Stars API has no endpoint that lists maps, so like the
brawler portraits (see brawlers.py) this comes from Brawlify's public API
(api.brawlapi.com, no key needed):

    python -m src.maps fetch

writes data/maps.json as a list of {"id", "name", "mode", "image"} objects,
keeping only maps that are currently in the game (`disabled: false`) and
belong to the 3v3 draft modes in RANKED_MODES. Brawlify lists ~1300 maps
including every retired one and all Showdown/event modes; the picker only
needs the ones that can show up in a Ranked draft.

Checked against real battlelog data (2026-09-19):
  - RANKED_MODES: see the comment on it. A missing mode can still be typed in
    as a free-text map name; wrong extras only add clutter (the picker has a
    mode filter).
  - Brawlify's map-name spelling differs from the battlelog's `event.map` in
    places ("Belles Rock" vs. "Belle's Rock", "Out In The Open" vs. "Out in the
    Open"), and the trained model only knows the battlelog's. /maps therefore
    reconciles this list with the crawled names and lets the battlelog spelling
    win; maps the crawl saw that Brawlify lacks are added (see api.py).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import requests

from .brawlers import drop_dead_images

log = logging.getLogger("maps")

DEFAULT_PATH = "data/maps.json"
BRAWLIFY_MAPS_URL = "https://api.brawlapi.com/v1/maps"

# Measured on real data (2026-09-19, 159 soloRanked battles): exactly these six
# modes occur in Ranked; Wipeout, which I first assumed, never did (it shows up
# only in ordinary trophy battles). Re-check with `python -m src.extract types`.
RANKED_MODES: list[str] = ["Gem Grab", "Brawl Ball", "Knockout", "Hot Zone", "Heist", "Bounty"]


def load_map_details(path: str = DEFAULT_PATH) -> list[dict[str, Any]]:
    """[{"name": str, "mode": str | None, "image": str | None}, ...] - empty if
    `python -m src.maps fetch` hasn't been run yet."""
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def fetch_and_save(path: str = DEFAULT_PATH, modes: list[str] | None = None) -> list[dict[str, Any]]:
    modes = modes or RANKED_MODES
    mode_order = {m: i for i, m in enumerate(modes)}

    resp = requests.get(BRAWLIFY_MAPS_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("list", []) if isinstance(data, dict) else data

    picked: dict[tuple[str, str], dict[str, Any]] = {}
    for m in items:
        mode = (m.get("gameMode") or {}).get("name")
        if m.get("disabled") or mode not in mode_order:
            continue
        key = (mode, m["name"])
        # Brawlify sometimes lists the same map twice under two ids; keep the newest.
        if key not in picked or m["id"] > picked[key]["id"]:
            picked[key] = {"id": m["id"], "name": m["name"], "mode": mode, "image": m.get("imageUrl")}

    entries = sorted(picked.values(), key=lambda e: (mode_order[e["mode"]], e["name"]))
    if not entries:
        raise RuntimeError(f"No active maps found for modes {modes} - has Brawlify's data format changed?")

    drop_dead_images(entries)
    missing = [e["name"] for e in entries if not e["image"]]
    if missing:
        log.warning("No working image for %d maps (the UI shows initials instead): %s", len(missing), missing)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    per_mode = {m: sum(1 for e in entries if e["mode"] == m) for m in modes}
    log.info("Wrote %d maps to %s: %s", len(entries), path, per_mode)
    return entries


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Fetch the map list + images (Brawlify) into data/maps.json.")
    parser.add_argument("cmd", nargs="?", default="fetch", choices=["fetch"])
    parser.add_argument("--modes", nargs="+", default=None, help=f"Game modes to include (default: {RANKED_MODES})")
    args = parser.parse_args()
    fetch_and_save(modes=args.modes)
