"""
Thin wrapper around the official Brawl Stars API.

Docs: https://developer.brawlstars.com/#/documentation
Auth: Bearer token, tied to the public IP you generated it for. If your IP
      changes (dynamic home connection, new server), requests start failing
      with 403 and you need to generate a new key for the new IP.
"""
from __future__ import annotations

import os
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Optional

import requests
from dotenv import load_dotenv

BASE_URL = "https://api.brawlstars.com/v1"


class BrawlStarsAPIError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"[{status_code}] {message}")


@dataclass
class RateLimiter:
    """Simple limiter: at most `max_per_second` calls/sec. Tune down if you
    see repeated 429s - the exact current limit isn't published, so this
    errs conservative by default."""
    max_per_second: float = 8.0
    _last_call: float = field(default=0.0, init=False)

    def wait(self) -> None:
        min_interval = 1.0 / self.max_per_second
        elapsed = time.monotonic() - self._last_call
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call = time.monotonic()


class BrawlStarsClient:
    def __init__(self, token: Optional[str] = None, max_per_second: float = 8.0):
        load_dotenv()  # the README tells you to put BS_API_TOKEN in .env - nothing else reads that file
        # The key only works from the IP it was created for. From a host without a
        # fixed IP (Railway Hobby, a home connection that changes), create the key
        # for RoyaleAPI's proxy IP 45.79.218.79 and set
        # BS_API_BASE_URL=https://bsproxy.royaleapi.dev/v1 - the proxy forwards to the official API.
        self.base_url = os.environ.get("BS_API_BASE_URL", BASE_URL).rstrip("/")
        self.token = token or os.environ.get("BS_API_TOKEN")
        if not self.token:
            raise RuntimeError(
                "No API token found. Set BS_API_TOKEN in your environment or .env file."
            )
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {self.token}", "Accept": "application/json"}
        )
        self.limiter = RateLimiter(max_per_second)

    @staticmethod
    def _encode_tag(tag: str) -> str:
        tag = tag.strip()
        if not tag.startswith("#"):
            tag = "#" + tag
        return urllib.parse.quote(tag)

    def _get(self, path: str, params: Optional[dict] = None, retries: int = 3) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        for attempt in range(retries):
            self.limiter.wait()
            resp = self.session.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 403:
                raise BrawlStarsAPIError(
                    403,
                    "Forbidden - your API key's whitelisted IP probably doesn't match "
                    "your current public IP anymore. Generate a new key at "
                    "developer.brawlstars.com.",
                )
            raise BrawlStarsAPIError(resp.status_code, resp.text)
        raise BrawlStarsAPIError(429, "Rate limited repeatedly, giving up.")

    def get_player(self, tag: str) -> dict[str, Any]:
        return self._get(f"/players/{self._encode_tag(tag)}")

    def get_battlelog(self, tag: str) -> list[dict[str, Any]]:
        data = self._get(f"/players/{self._encode_tag(tag)}/battlelog")
        return data.get("items", [])

    def get_club_members(self, club_tag: str) -> list[dict[str, Any]]:
        data = self._get(f"/clubs/{self._encode_tag(club_tag)}/members")
        return data.get("items", [])

    def get_rankings(self, kind: str, country: str = "global", limit: int = 200) -> list[dict[str, Any]]:
        """kind: 'players' or 'clubs'."""
        data = self._get(f"/rankings/{country}/{kind}", params={"limit": limit})
        return data.get("items", [])

    def get_brawlers(self) -> list[dict[str, Any]]:
        data = self._get("/brawlers")
        return data.get("items", [])
