"""Immutable-style draft state: map, bans/picks so far for both teams, whose
turn it is next. The turn sequence comes from DraftRules (rules.py)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .rules import DraftRules, Phase, Team, Turn


@dataclass
class DraftState:
    map_name: Optional[str]
    rules: DraftRules = field(default_factory=DraftRules)
    bans: dict[Team, list[str]] = field(default_factory=lambda: {Team.A: [], Team.B: []})
    picks: dict[Team, list[str]] = field(default_factory=lambda: {Team.A: [], Team.B: []})

    def __post_init__(self) -> None:
        self._turns = self.rules.turn_order()

    @property
    def ply(self) -> int:
        return (
            len(self.bans[Team.A]) + len(self.bans[Team.B])
            + len(self.picks[Team.A]) + len(self.picks[Team.B])
        )

    def is_terminal(self) -> bool:
        return self.ply >= len(self._turns)

    def current_turn(self) -> Optional[Turn]:
        if self.is_terminal():
            return None
        return self._turns[self.ply]

    def used_brawlers(self) -> set[str]:
        return (
            set(self.bans[Team.A]) | set(self.bans[Team.B])
            | set(self.picks[Team.A]) | set(self.picks[Team.B])
        )

    def available_brawlers(self, pool: list[str]) -> list[str]:
        used = self.used_brawlers()
        return [b for b in pool if b not in used]

    def apply(self, brawler: str) -> "DraftState":
        """Returns a NEW state with `brawler` applied to whoever's turn it
        currently is - used by the search so branches don't mutate shared state."""
        turn = self.current_turn()
        if turn is None:
            raise ValueError("Draft already complete - no turn to apply a move to.")
        new = DraftState(
            map_name=self.map_name,
            rules=self.rules,
            bans={t: list(v) for t, v in self.bans.items()},
            picks={t: list(v) for t, v in self.picks.items()},
        )
        target = new.bans if turn.phase is Phase.BAN else new.picks
        target[turn.team].append(brawler)
        return new
