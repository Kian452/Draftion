"""
Ranked-Draft turn sequence (Abschnitt 3, 9).

Confirmed (from community draft tools, per the spec): 3 bans per team, 6 total.

Confirmed by Kian from playing:
  - Ban phase: two blocks - his whole team (Team A) bans all its brawlers
    first, then the other team (Team B) bans all of its. Not alternating.
  - Pick phase: 1-2-2-1 - the team with first pick takes 1 brawler, the other
    team takes 2, the first team takes 2, the other team takes the last 1.
    (The spec's "alternating single picks, no snake" was wrong about this.)
  - Which team has first pick is decided per match, so it is not fixed here:
    `pick_first` is a parameter and the frontend lets you choose it before
    each draft (GET /rules?first_pick=A|B, `first_pick` in POST /recommend).

Not yet checked against the live client (Abschnitt 3, 9): whether the enemy
team also bans first when they have first pick - the ban block order above is
currently the same either way.

The whole sequence is expressed here as one small configurable turn list so it
can be corrected in one place - everything downstream (DraftState, search, the
frontend via GET /rules) just consumes `DraftRules.turn_order()`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Phase(Enum):
    BAN = "ban"
    PICK = "pick"


class Team(Enum):
    A = 0
    B = 1

    def other(self) -> "Team":
        return Team.B if self is Team.A else Team.A


@dataclass(frozen=True)
class Turn:
    phase: Phase
    team: Team


@dataclass(frozen=True)
class DraftRules:
    bans_per_team: int = 3
    picks_per_team: int = 3
    ban_first: Team = Team.A
    pick_first: Team = Team.A
    # Pick phase as blocks of consecutive picks, teams alternating per block and
    # starting with `pick_first`. (1, 2, 2, 1) = first team 1, other team 2,
    # first team 2, other team 1. (1, 1, 1, 1, 1, 1) would be strictly alternating.
    pick_pattern: tuple[int, ...] = (1, 2, 2, 1)

    def __post_init__(self) -> None:
        first_total, second_total = sum(self.pick_pattern[0::2]), sum(self.pick_pattern[1::2])
        if first_total != self.picks_per_team or second_total != self.picks_per_team:
            raise ValueError(
                f"pick_pattern {self.pick_pattern} gives the teams {first_total} and {second_total} "
                f"picks, but picks_per_team is {self.picks_per_team}"
            )

    def turn_order(self) -> list[Turn]:
        # Ban phase: one team's whole block, then the other's.
        bans = (
            [Turn(Phase.BAN, self.ban_first)] * self.bans_per_team
            + [Turn(Phase.BAN, self.ban_first.other())] * self.bans_per_team
        )

        picks: list[Turn] = []
        team = self.pick_first
        for block in self.pick_pattern:
            picks += [Turn(Phase.PICK, team)] * block
            team = team.other()

        return bans + picks

    def total_turns(self) -> int:
        return 2 * (self.bans_per_team + self.picks_per_team)


DEFAULT_RULES = DraftRules()
