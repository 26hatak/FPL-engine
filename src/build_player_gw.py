"""Build data/processed/player_gw.parquet from player_match + gameweeks.

Run from the project root: python src/build_player_gw.py

player_gw: one row per (season, player_code, gw).
  - outcome columns (targets): what happened in that GW, doubles summed
  - f_* columns (features): only from matches in EARLIER gameweeks
  - snap_* columns (features): the latest daily snapshot before the deadline

Tests, including the leakage test: `pytest tests/test_player_gw.py -q`.
"""
from pathlib import Path

import pandas as pd

from build_player_match import OUTCOMES

PROCESSED = Path("data/processed")
GW_KEYS = ["season", "player_code", "gw"]


def build_gw_grid(player_match: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, player_code, gw): every GW from the player's FIRST
    appearance in that season to the LAST gw of that season in player_match.

    Why a grid: a blank gameweek (the player's team has no fixture) has no
    match row at all. Without the grid the row simply vanishes, and "last 3
    gameweeks" quietly becomes "last 3 rows", which spans a different time.

    Returns only the GW_KEYS columns, no duplicates, sorted by GW_KEYS.

    Assumes no blank in a player's first gameweek (GW1 never has blanks), since
    the start is inferred from the first row.
    """
    first = player_match.groupby(["season", "player_code"])["gw"].min().rename("first_gw")  # per player
    last = player_match.groupby("season")["gw"].max().rename("last_gw")                     # per season
    spans = first.reset_index().merge(last.reset_index(), on="season")

    # One list per player-season (~2k), then one row per list item.
    spans["gw"] = [list(range(a, b + 1)) for a, b in zip(spans["first_gw"], spans["last_gw"])]
    grid = spans.explode("gw").astype({"gw": int})
    return grid[GW_KEYS].sort_values(GW_KEYS).reset_index(drop=True)


def aggregate_targets(player_match: pd.DataFrame) -> pd.DataFrame:
    """Collapse matches to gameweeks: one row per GW_KEYS with
      - every OUTCOMES column summed (a double gameweek counts both matches)
      - n_fixtures: number of matches the player had that GW
      - value: price at that GW (first match's value; same for both in a double)
      - n_home: home matches that GW (0, 1 or 2): a double can be one of each
      - team_code, position: the player's club and position that GW
      - fpl_xp: FPL's own expected points (per GW already, so taken once)

    Opponents stay out on purpose: a double has two, and one GW row can't hold
    both without averaging them away. Phase 1 predicts per fixture instead
    (see DATA.md, "Double gameweeks").

    Trap: pandas .sum() turns all-NaN into 0. defensive_contribution is NaN
    for 2024-25 (not recorded) and must STAY NaN, hence min_count=1.
    """
    by_gw = player_match.groupby(GW_KEYS)
    targets = by_gw[OUTCOMES].sum(min_count=1)
    targets["n_fixtures"] = by_gw.size()
    targets["n_home"] = by_gw["was_home"].sum()
    targets["value"] = by_gw["value"].first()
    targets["team_code"] = by_gw["team_code"].first()
    targets["position"] = by_gw["position"].first()
    targets["fpl_xp"] = by_gw["fpl_xp"].first()
    return targets.reset_index()


def add_rolling_features(player_gw: pd.DataFrame, windows=(3, 5)) -> pd.DataFrame:
    """Add f_* feature columns, per player, using ONLY earlier gameweeks.

    For each N in windows, over the player's last N gameweeks WITH a match
    (n_fixtures > 0), across season boundaries, using as many as exist up to N
    (min_periods=1):
      f_minutes_mean_{N}   mean minutes per gameweek
      f_start_rate_{N}     starts / fixtures
      f_xg_per90_{N}       sum(expected_goals) / sum(minutes) * 90 (NaN if 0 minutes)
      f_xa_per90_{N}       same with expected_assists
      f_defcon_per90_{N}   same with defensive_contribution, but count only the
                           minutes from gameweeks where DefCon was recorded
                           (2024-25 minutes must not dilute the rate)

    The leakage rule: the row for GW t may only use GWs < t. A player's first
    ever row therefore has NaN features. A blank-GW row carries the values
    from the player's last match before it.

    How: rolling sums on the played rows only give the state at the END of
    each GW. Put them back on the grid, forward-fill per player (blanks keep
    the last state), then shift(1) per player so each GW sees only the state
    at the end of the previous GW.

    Returns player_gw with the f_* columns added, rows in the same order.
    """
    # Time order per player, across seasons ("2024-25" < "2025-26" as strings).
    df = player_gw.sort_values(["player_code", "season", "gw"])
    played = df[df["n_fixtures"] > 0].copy()
    # DefCon rate: only minutes from GWs where DefCon was recorded count.
    played["defcon_minutes"] = played["minutes"].where(played["defensive_contribution"].notna())
    by_player = played.groupby("player_code")

    def rolling_sum(col, n):
        # NaN values are skipped; an all-NaN window stays NaN.
        return by_player[col].rolling(n, min_periods=1).sum().reset_index(level=0, drop=True)

    def per90(stat, minutes):
        return (stat / minutes * 90).where(minutes > 0)

    features = pd.DataFrame(index=played.index)
    for n in windows:
        minutes = rolling_sum("minutes", n)
        features[f"f_minutes_mean_{n}"] = (
            by_player["minutes"].rolling(n, min_periods=1).mean().reset_index(level=0, drop=True))
        features[f"f_start_rate_{n}"] = rolling_sum("starts", n) / rolling_sum("n_fixtures", n)
        features[f"f_xg_per90_{n}"] = per90(rolling_sum("expected_goals", n), minutes)
        features[f"f_xa_per90_{n}"] = per90(rolling_sum("expected_assists", n), minutes)
        features[f"f_defcon_per90_{n}"] = per90(rolling_sum("defensive_contribution", n),
                                                rolling_sum("defcon_minutes", n))

    cols = list(features.columns)
    df = df.join(features)                                 # NaN on blank rows
    df[cols] = df.groupby("player_code")[cols].ffill()     # blanks carry the last state
    df[cols] = df.groupby("player_code")[cols].shift(1)    # GW t sees only the end of t-1
    return df.loc[player_gw.index]


SNAPSHOT_COLUMNS = {
    "snapshot_time": "snap_time",
    "status": "snap_status",                                  # a / d / i / s / u
    "chance_of_playing_next_round": "snap_chance_of_playing",  # NaN = no news
    "news": "snap_news",
    "now_cost": "snap_price",                                  # tenths of £m
    "selected_by_percent": "snap_selected_pct",
    "ep_next": "snap_ep_next",                                 # FPL's own prediction: the baseline
}


def add_snapshot_features(player_gw: pd.DataFrame, players_daily: pd.DataFrame) -> pd.DataFrame:
    """Attach what was known at each deadline: for every (season, player, gw),
    the LATEST snapshot taken STRICTLY BEFORE that gameweek's deadline.

    An as-of join (merge_asof, direction="backward"): a snapshot from after the
    deadline would leak, e.g. ep_next switches to the following gameweek the
    moment a deadline passes. A snapshot taken exactly at the deadline is
    excluded too (allow_exact_matches=False). Rows with no earlier snapshot
    (past seasons, and 2026-27 before Oct 7) get NaN.
    """
    snaps = (players_daily.rename(columns=SNAPSHOT_COLUMNS)
             [["season", "player_code", *SNAPSHOT_COLUMNS.values()]]
             .sort_values("snap_time"))
    left = player_gw.assign(_row=range(len(player_gw))).sort_values("deadline_time")
    merged = pd.merge_asof(left, snaps, left_on="deadline_time", right_on="snap_time",
                           by=["season", "player_code"], direction="backward",
                           allow_exact_matches=False)
    merged = merged.sort_values("_row").drop(columns="_row")
    merged.index = player_gw.index
    return merged


def build(player_match: pd.DataFrame, gameweeks: pd.DataFrame,
          players_daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Grid + targets + deadline + features. Blank GWs keep NaN outcomes
    (no match happened) and n_fixtures = n_home = 0; filter on n_fixtures > 0
    to train. A blank still has a club and position: forward-filled from the
    player's last match, never back-filled (that would copy a later transfer
    into the past)."""
    grid = build_gw_grid(player_match)
    player_gw = grid.merge(aggregate_targets(player_match), on=GW_KEYS, how="left")
    for col in ["n_fixtures", "n_home"]:
        player_gw[col] = player_gw[col].fillna(0).astype(int)
    # The grid is sorted by season, player, gw, so within a player rows run in time order.
    carried = ["team_code", "position"]
    player_gw[carried] = player_gw.groupby("player_code")[carried].ffill()
    player_gw["team_code"] = player_gw["team_code"].astype(int)
    player_gw = player_gw.merge(gameweeks, on=["season", "gw"], how="left")
    player_gw = add_rolling_features(player_gw)
    if players_daily is not None:
        player_gw = add_snapshot_features(player_gw, players_daily)
    return player_gw


def main() -> None:
    player_match = pd.read_parquet(PROCESSED / "player_match.parquet")
    gameweeks = pd.read_parquet(PROCESSED / "gameweeks.parquet")
    players_daily = pd.read_parquet(PROCESSED / "players_daily.parquet")
    player_gw = build(player_match, gameweeks, players_daily)
    player_gw.to_parquet(PROCESSED / "player_gw.parquet", index=False)
    print(f"player_gw {player_gw.shape}")


if __name__ == "__main__":
    main()
