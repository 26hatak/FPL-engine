"""Import a real FPL team by its public team ID (read-only, no password).

Run from the project root:
    python src/fpl_sync.py 123456            # prints the squad, bank, free transfers
    python src/fpl_sync.py 123456 --processed /path/to/data/processed

Why this exists: the transfer planner (Phase 3) must start from the user's
*actual* squad, bank and free transfers. The FPL app shows those after login,
but we never ask for a password. Everything here comes from the public,
unofficial API that anyone can read for any team ID:

    entry/{id}/              team name, started_event, current_event
    entry/{id}/history/      per GW: bank, value, event_transfers(_cost); chips used
    entry/{id}/transfers/    every transfer: element_in/out, *_cost, event
    entry/{id}/event/{gw}/picks/   the 15 players at that GW's deadline, active_chip

What the public API does NOT give is the selling price of each player (only the
logged-in `my-team/` endpoint does), nor the number of free transfers. Both are
therefore *reconstructed* by replaying the history with the game's rules.

Design: HTTP lives in `FplClient` / `fetch_*`; every rule lives in pure
`reconstruct_*` / `selling_price` functions that take plain dicts and
DataFrames, so tests/test_fpl_sync.py checks them offline with hand-made
histories.

2026/27 rules used here, read on https://fantasy.premierleague.com/help/rules
on 2026-11-22 (and the matching `game_settings` in bootstrap-static):
  - "Unlimited transfers can be made at no cost until your first deadline.
    After your first deadline you will receive 1 free transfer each Gameweek."
    Each extra transfer costs 4 points.
  - Unused free transfers carry over; "The maximum number of free transfers you
    can store in any Gameweek is 5." (game_settings.max_extra_free_transfers = 4,
    i.e. 1 + 4 banked.)
  - "When playing either a Wildcard or your Free Hit chip, any saved free
    transfers are retained for the following Gameweek. If you had 2 saved free
    transfers, you will still have 2 ..." -> no +1 in a chip week.
  - Free Hit: "At the next deadline your squad is returned to how it was at the
    start of the Gameweek."
  - Selling price: "you keep half of the increase when selling the player,
    rounded down to the nearest £0.1m" (7.5 -> 7.8 sells for 7.6).
    game_settings.transfers_sell_on_fee = 0.5. A fall is taken in full.
  - Chips come in two sets: first set until the GW19 deadline, second set
    after it. Wildcard and Free Hit only from the second GW of your season.

Verified by hand against a real team on 2026-11-22: transfers made in a
wildcard or free-hit week ARE listed in `transfers/` (27 and 7 of them), while
`history` reports event_transfers = 0 for those weeks. So free-hit transfers
must be ignored when working out purchase prices (they were reverted), and
`element_out_cost` is the money actually received (bank deltas match it).

Known approximations (flagged in `TeamState.warnings` / `price_source`):
  - Players in the very first squad have no transfer record. Their purchase
    price is taken as the price at the first deadline (`player_gw.value` for
    `started_event`). Exact for teams started in GW1 (prices don't move before
    the season); a mid-season starter may be off by a price change or two.
  - Current price = the latest `now_cost` in players_daily (the daily snapshot),
    so it can be up to a day old; `prices_as_of` says how old.
  - Transfers for the upcoming GW are applied to the squad ("pending"). If the
    user has a Free Hit lined up for that GW we can't see it publicly, so those
    pending transfers would wrongly look permanent.
  - Special free-transfer top-ups FPL sometimes grants (e.g. 2024/25 AFCON) are
    not in the rules; if one happens, the replay disagrees with the hit costs
    FPL charged and a warning is added.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import requests

from app_data import load_table
from build_player_match import CURRENT_SEASON

API = "https://fantasy.premierleague.com/api/"
PROCESSED = Path("data/processed")

# Politeness: one request at a time, ~1 s apart, a generic browser-like
# User-Agent (no personal contact details), and a timeout so a hung API
# can't hang the app.
USER_AGENT = "Mozilla/5.0 (compatible; fpl-engine/0.1; read-only team import)"
TIMEOUT = 15     # seconds per request
PAUSE = 1.0      # seconds between requests

MAX_FREE_TRANSFERS = 5     # rules: "maximum ... you can store in any Gameweek is 5"
HIT_COST = 4               # points per transfer beyond the free ones
SELL_ON_FEE = 0.5          # you keep half of a rise
CHIP_RESET_FT = {"wildcard", "freehit"}   # chips that keep saved FTs unchanged
# (first_gw, last_gw) each chip can be played in, per set (bootstrap-static `chips`).
CHIP_WINDOWS = {
    "wildcard": [(2, 19), (20, 38)],
    "freehit": [(2, 19), (20, 38)],
    "bboost": [(1, 19), (20, 38)],
    "3xc": [(1, 19), (20, 38)],
}


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

@dataclass
class SquadPlayer:
    player_code: int        # stable FPL `code`: the key the rest of the repo uses
    element: int            # this season's id, kept only to talk to the API
    web_name: str
    position: str           # GK / DEF / MID / FWD
    team_code: int
    club: str               # short name, e.g. ARS
    purchase_price: int     # tenths of £m
    current_price: int      # tenths of £m (latest players_daily now_cost)
    selling_price: int      # tenths of £m: what selling him would add to the bank
    price_source: str       # "transfer" | "first_deadline" | "current_price_fallback"


@dataclass
class TeamState:
    entry_id: int
    team_name: str
    as_of_gw: int                # squad = picks at this GW's deadline (+ pending transfers)
    next_gw: int | None          # the deadline the planner is planning for
    bank: int                    # tenths of £m, after pending transfers
    free_transfers: int          # for next_gw, before pending transfers
    free_transfers_left: int     # for next_gw, after pending transfers
    pending_transfers: list[dict]
    chips_used: list[dict]       # [{"name": "wildcard", "event": 4}, ...]
    chips_available: list[str]   # chips still playable in next_gw's half of the season
    squad: list[SquadPlayer]
    prices_as_of: str | None     # snapshot time of current_price
    warnings: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(p) for p in self.squad])

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# HTTP (the only part that touches the network)
# --------------------------------------------------------------------------- #

class FplApiError(RuntimeError):
    """The API is down, being updated, or returned something unexpected."""


class TeamNotFound(FplApiError):
    """No FPL team has this ID."""


class FplClient:
    """A tiny polite HTTP client: one request at a time with a pause between
    them, a timeout, and errors a user can read. Tests replace it with a fake
    that returns canned JSON, so they never hit the internet."""

    def __init__(self, base: str = API, pause: float = PAUSE, timeout: float = TIMEOUT):
        self.base = base
        self.pause = pause
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._last = 0.0

    def get(self, path: str):
        wait = self.pause - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        url = self.base + path
        try:
            resp = self.session.get(url, timeout=self.timeout)
        except requests.RequestException as exc:
            raise FplApiError(f"Could not reach the FPL API ({exc.__class__.__name__}). "
                              "Check your connection or try again later.") from exc
        finally:
            self._last = time.monotonic()
        if resp.status_code == 404:
            raise TeamNotFound(f"Not found: {url}")
        if resp.status_code == 503:
            raise FplApiError("The FPL API is unavailable (usually 'the game is being "
                              "updated' around a deadline). Try again in a few minutes.")
        if resp.status_code != 200:
            raise FplApiError(f"FPL API returned HTTP {resp.status_code} for {url}")
        try:
            return resp.json()
        except ValueError as exc:
            raise FplApiError(f"FPL API returned non-JSON for {url}") from exc


def fetch_entry(client: FplClient, entry_id: int) -> dict:
    try:
        return client.get(f"entry/{entry_id}/")
    except TeamNotFound:
        raise TeamNotFound(f"No FPL team with ID {entry_id}. The ID is the number in "
                           "the URL of your Points page: .../entry/<ID>/event/<GW>") from None


def fetch_history(client: FplClient, entry_id: int) -> dict:
    return client.get(f"entry/{entry_id}/history/")


def fetch_transfers(client: FplClient, entry_id: int) -> list[dict]:
    return client.get(f"entry/{entry_id}/transfers/")


def fetch_picks(client: FplClient, entry_id: int, gw: int) -> dict:
    return client.get(f"entry/{entry_id}/event/{gw}/picks/")


def fetch_team(entry_id: int, client: FplClient | None = None) -> dict:
    """All raw JSON needed for one team: 4 requests (5 if the latest GW was a
    free hit, since the real squad is then the one from the GW before)."""
    client = client or FplClient()
    entry = fetch_entry(client, entry_id)
    gw = entry.get("current_event")
    if not gw or (entry.get("started_event") or 0) > gw:
        raise FplApiError(f"Team {entry_id} hasn't reached its first deadline yet, so "
                          "its squad isn't public. Try again after the next deadline.")
    raw = {"entry": entry,
           "history": fetch_history(client, entry_id),
           "transfers": fetch_transfers(client, entry_id),
           "picks": fetch_picks(client, entry_id, gw),
           "prev_picks": None}
    if raw["picks"].get("active_chip") == "freehit":
        raw["prev_picks"] = fetch_picks(client, entry_id, gw - 1)
    return raw


# --------------------------------------------------------------------------- #
# Pure rules (tested offline)
# --------------------------------------------------------------------------- #

def selling_price(purchase: int, current: int) -> int:
    """Selling price in tenths of £m.

    A rise: you keep half of it, rounded DOWN to £0.1m. In tenths that's
    integer division: 7.5 -> 7.8 is a rise of 3 tenths, 3 // 2 = 1, sells at 7.6.
    A fall: you take all of it (sell at the current price).
    Integers on purpose: floats would make 0.1 + 0.2 != 0.3 and break the
    rounding exactly at the boundaries.
    """
    rise = current - purchase
    if rise <= 0:
        return current
    return purchase + int(rise * SELL_ON_FEE)   # rise >= 0, so int() floors


def chip_events(chips: list[dict], name: str | None = None) -> dict[int, str]:
    """{event: chip name} from history['chips'] (optionally one chip only)."""
    return {c["event"]: c["name"] for c in chips if name is None or c["name"] == name}


def reconstruct_free_transfers(history: list[dict], chips: list[dict],
                               started_event: int) -> tuple[int, list[str]]:
    """Free transfers available for the GW after the last row of `history`.

    Replay, one GW at a time (ft[g] = free transfers at GW g's deadline):
      - before the first deadline (g = started_event) transfers are unlimited,
        so the first real value is ft[started_event + 1] = 1;
      - wildcard / free hit in g: ft[g+1] = ft[g]   (saved FTs kept, no +1);
      - otherwise:                ft[g+1] = min(5, max(ft[g] - made[g], 0) + 1).
    `max(.., 0)` because extra transfers are paid with points, not borrowed
    from next week.

    Returns (ft for the next GW, warnings). As a check, each normal week's
    expected hit cost 4 * max(made - ft, 0) is compared with the
    event_transfers_cost FPL actually charged; a mismatch means the replay is
    off (e.g. FPL handed out bonus transfers) and is reported, not hidden.
    """
    chip_by_gw = chip_events(chips)
    warnings = []
    ft = None   # unlimited before the first deadline
    for row in sorted(history, key=lambda r: r["event"]):
        g, made = row["event"], row["event_transfers"]
        if g < started_event:
            continue
        if ft is None:          # first GW of the team's season
            ft = 1
            continue
        if chip_by_gw.get(g) in CHIP_RESET_FT:
            continue            # ft unchanged
        expected_cost = HIT_COST * max(made - ft, 0)
        if expected_cost != row["event_transfers_cost"]:
            warnings.append(f"GW{g}: replay expects a {expected_cost}-point hit for "
                            f"{made} transfers with {ft} free, FPL charged "
                            f"{row['event_transfers_cost']}. Free transfers may be off.")
        ft = min(MAX_FREE_TRANSFERS, max(ft - made, 0) + 1)
    return (ft if ft is not None else 1), warnings


def reconstruct_squad_elements(picks: dict, prev_picks: dict | None,
                               transfers: list[dict], as_of_gw: int
                               ) -> tuple[list[int], list[dict]]:
    """The 15 elements the user owns now, and the pending transfers applied.

    Start from the picks at the latest deadline. If that GW was a free hit,
    the squad snaps back to the start of that GW, i.e. the previous GW's picks.
    Then apply transfers already made for the NEXT deadline (event > as_of_gw):
    they're in `transfers/` but not yet in any picks.
    """
    base = prev_picks if picks.get("active_chip") == "freehit" else picks
    if base is None:
        raise ValueError("latest GW was a free hit: prev_picks (GW before) is required")
    squad = [p["element"] for p in base["picks"]]
    pending = sorted((t for t in transfers if t["event"] > as_of_gw), key=lambda t: t["time"])
    for t in pending:
        if t["element_out"] not in squad:
            raise ValueError(f"pending transfer sells {t['element_out']}, not in the squad")
        squad[squad.index(t["element_out"])] = t["element_in"]
    if len(squad) != 15 or len(set(squad)) != 15:
        raise ValueError(f"expected 15 distinct players, got {squad}")
    return squad, pending


def reconstruct_purchase_prices(squad: list[int], transfers: list[dict], chips: list[dict],
                                first_deadline_price: dict[int, int]
                                ) -> dict[int, tuple[int, str]]:
    """{element: (purchase price, source)} for the current squad.

    Purchase price = element_in_cost of the LATEST transfer that brought him
    in (a player sold and re-bought is bought at the new price). Transfers in
    free-hit weeks are skipped: that squad was reverted, so those purchases
    never happened as far as the real squad is concerned. A player with no
    transfer-in has been there since the first squad: use the price at the
    first deadline (see module docstring for why that's an approximation).
    """
    freehit_gws = set(chip_events(chips, "freehit"))
    bought = {}
    for t in sorted(transfers, key=lambda t: t["time"]):
        if t["event"] not in freehit_gws:
            bought[t["element_in"]] = t["element_in_cost"]
    out = {}
    for e in squad:
        if e in bought:
            out[e] = (bought[e], "transfer")
        elif e in first_deadline_price:
            out[e] = (first_deadline_price[e], "first_deadline")
        else:
            out[e] = (None, "current_price_fallback")
    return out


def chips_available(chips_used: list[dict], next_gw: int | None) -> list[str]:
    """Chips the user can still play at next_gw.

    Each chip comes once per half (GW1-19, GW20-38): the first-half chip is
    gone if it was used, or simply expires, at the GW19 deadline. Free Hit
    can't be played in two consecutive GWs.
    """
    if next_gw is None:
        return []
    half = 0 if next_gw <= 19 else 1
    half_gws = range(1, 20) if half == 0 else range(20, 39)
    left = []
    for name, windows in CHIP_WINDOWS.items():
        lo, hi = windows[half]
        if not lo <= next_gw <= hi:
            continue
        if any(c["name"] == name and c["event"] in half_gws for c in chips_used):
            continue
        if name == "freehit" and any(c["name"] == "freehit" and c["event"] == next_gw - 1
                                     for c in chips_used):
            continue
        left.append(name)
    return left


def reconstruct_team(raw: dict, players: pd.DataFrame, teams: pd.DataFrame,
                     prices: pd.DataFrame, first_deadline_values: pd.DataFrame,
                     season: str = CURRENT_SEASON) -> TeamState:
    """Turn the raw API JSON into a TeamState. Pure: no network, no disk.

    players:  players table (season, element, player_code, web_name, team_code, position)
    teams:    teams table (season, team_code, team_short)
    prices:   players_daily rows (season, player_code, snapshot_time, now_cost)
    first_deadline_values: player_gw rows (season, player_code, gw, value)
    """
    entry, hist = raw["entry"], raw["history"]
    as_of = entry["current_event"]
    started = entry.get("started_event") or 1
    warnings = []

    squad, pending = reconstruct_squad_elements(raw["picks"], raw["prev_picks"],
                                                raw["transfers"], as_of)

    # element -> player_code etc. (`element` is per season, so map once, here)
    lookup = players[players["season"] == season].set_index("element")
    missing = [e for e in squad if e not in lookup.index]
    if missing:
        raise ValueError(f"elements {missing} not in the {season} players table; "
                         "rebuild data (python src/build_all.py)")
    codes = {e: int(lookup.at[e, "player_code"]) for e in squad}

    latest = (prices[prices["season"] == season].sort_values("snapshot_time")
              .groupby("player_code").tail(1).set_index("player_code"))
    prices_as_of = str(latest["snapshot_time"].max()) if len(latest) else None

    fdv = first_deadline_values[(first_deadline_values["season"] == season)
                                & (first_deadline_values["gw"] == started)]
    fdv = fdv.dropna(subset=["value"]).set_index("player_code")["value"]
    first_price = {e: int(fdv[c]) for e, c in codes.items() if c in fdv.index}
    purchase = reconstruct_purchase_prices(squad, raw["transfers"], hist["chips"], first_price)

    club = teams[teams["season"] == season].set_index("team_code")["team_short"]
    out = []
    for e in squad:
        p = lookup.loc[e]
        code = codes[e]
        if code not in latest.index:
            raise ValueError(f"no price for player_code {code} in players_daily")
        now = int(latest.at[code, "now_cost"])
        buy, source = purchase[e]
        if buy is None:
            buy = now
            warnings.append(f"{p['web_name']}: purchase price unknown, assumed = current price")
        out.append(SquadPlayer(player_code=code, element=int(e), web_name=p["web_name"],
                               position=p["position"], team_code=int(p["team_code"]),
                               club=str(club.get(p["team_code"], "?")),
                               purchase_price=int(buy), current_price=now,
                               selling_price=selling_price(int(buy), now),
                               price_source=source))

    rows = [r for r in hist["current"] if r["event"] <= as_of]
    ft, ft_warn = reconstruct_free_transfers(rows, hist["chips"], started)
    warnings += ft_warn
    # A free hit reverts money too: the real bank is the one from the GW before.
    bank_gw = as_of - 1 if raw["picks"].get("active_chip") == "freehit" else as_of
    bank = next(r["bank"] for r in rows if r["event"] == bank_gw)
    bank += sum(t["element_out_cost"] - t["element_in_cost"] for t in pending)

    next_gw = as_of + 1 if as_of < 38 else None
    chips_used = [{"name": c["name"], "event": c["event"]} for c in hist["chips"]]
    return TeamState(
        entry_id=int(entry["id"]), team_name=entry.get("name", ""), as_of_gw=as_of,
        next_gw=next_gw, bank=int(bank), free_transfers=ft,
        free_transfers_left=max(ft - len(pending), 0),
        pending_transfers=[{k: t[k] for k in ("event", "element_in", "element_in_cost",
                                              "element_out", "element_out_cost")}
                           for t in pending],
        chips_used=chips_used, chips_available=chips_available(chips_used, next_gw),
        squad=out, prices_as_of=prices_as_of, warnings=warnings)


# --------------------------------------------------------------------------- #
# Glue: fetch + local tables -> TeamState
# --------------------------------------------------------------------------- #

def load_tables(processed: Path = PROCESSED) -> dict[str, pd.DataFrame]:
    """The local tables reconstruct_team needs, via app_data.load_table: read
    from data/processed/ on a laptop, downloaded from the data-latest release
    when hosted (where data/processed/ doesn't exist). Only published tables
    are used, so a hosted sync can't fail on a missing file."""
    return {
        "players": load_table("players", processed),
        "teams": load_table("teams", processed),
        "prices": load_table("prices", processed),                    # latest snapshot
        "first_deadline_values": load_table("gw_values", processed),  # price per GW
    }


def sync_team(entry_id: int, processed: Path = PROCESSED,
              client: FplClient | None = None) -> TeamState:
    """Fetch team `entry_id` from the API and reconstruct its state."""
    raw = fetch_team(entry_id, client)
    return reconstruct_team(raw, **load_tables(processed))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("entry_id", type=int, help="FPL team ID")
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    args = ap.parse_args()
    try:
        team = sync_team(args.entry_id, args.processed)
    except FplApiError as exc:
        raise SystemExit(f"Error: {exc}")
    print(f"{team.team_name} (ID {team.entry_id}): squad as of GW{team.as_of_gw}, "
          f"planning for GW{team.next_gw}")
    print(f"bank £{team.bank / 10:.1f}m | free transfers {team.free_transfers} "
          f"({team.free_transfers_left} left after {len(team.pending_transfers)} pending) | "
          f"chips used {[(c['name'], c['event']) for c in team.chips_used]} | "
          f"available {team.chips_available} | prices as of {team.prices_as_of}")
    df = team.to_frame()
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    df = df.sort_values("position", key=lambda s: s.map(order))
    for col in ("purchase_price", "current_price", "selling_price"):
        df[col] = df[col] / 10
    print(df.drop(columns=["element"]).to_string(index=False))
    print(f"squad selling value £{df['selling_price'].sum():.1f}m "
          f"+ bank = £{df['selling_price'].sum() + team.bank / 10:.1f}m")
    for w in team.warnings:
        print("warning:", w)


if __name__ == "__main__":
    main()
