"""Build data/processed/player_gw.parquet from player_match + gameweeks.

Run from the project root: python src/build_player_gw.py

player_gw: one row per (season, player_code, gw).
  - outcome columns (targets): what happened in that GW, doubles summed
  - f_* columns (features): only from matches in EARLIER gameweeks

Three functions are left for you (marked YOUR TURN). Their tests are in
tests/test_player_gw.py, including a leakage test: `pytest tests/test_player_gw.py -q`.
"""
from pathlib import Path

import pandas as pd

from build_player_match import OUTCOMES

PROCESSED = Path("data/processed")
GW_KEYS = ["season", "player_code", "gw"]


# ---------------------------------------------------------------- YOUR TURN 3
def build_gw_grid(player_match: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, player_code, gw): every GW from the player's FIRST
    appearance in that season to the LAST gw of that season in player_match.

    Why a grid: a blank gameweek (the player's team has no fixture) has no
    match row at all. Without the grid the row simply vanishes, and "last 3
    gameweeks" quietly becomes "last 3 rows", which spans a different time.

    Returns only the GW_KEYS columns, no duplicates, sorted by GW_KEYS.

    Hint: per season take the gw range; per (season, player) take the first gw;
    a cross join then filter (merge(how="cross")) or a groupby + explode both work.
    """

    
    raise NotImplementedError("build_gw_grid: see docstring")


# ---------------------------------------------------------------- YOUR TURN 4
def aggregate_targets(player_match: pd.DataFrame) -> pd.DataFrame:
    """Collapse matches to gameweeks: one row per GW_KEYS with
      - every OUTCOMES column summed (a double gameweek counts both matches)
      - n_fixtures: number of matches the player had that GW
      - value: price at that GW (first match's value; same for both in a double)

    Trap: pandas .sum() turns all-NaN into 0. defensive_contribution is NaN
    for 2024-25 (not recorded) and must STAY NaN. Look up min_count.
    """
    raise NotImplementedError("aggregate_targets: see docstring")


# ---------------------------------------------------------------- YOUR TURN 5
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

    One clean way: compute rolling sums on the played rows only (that gives
    "state at the END of each GW"), put them back on the grid, forward-fill
    per player, then shift(1) per player so each GW sees only the state at
    the end of the previous GW. Sort by season, gw first!

    Returns player_gw with the f_* columns added, rows in the same order.
    """
    raise NotImplementedError("add_rolling_features: see docstring")


def build(player_match: pd.DataFrame, gameweeks: pd.DataFrame) -> pd.DataFrame:
    """Grid + targets + deadline + features. Blank GWs keep NaN outcomes
    (no match happened) and n_fixtures = 0; filter on n_fixtures > 0 to train."""
    grid = build_gw_grid(player_match)
    player_gw = grid.merge(aggregate_targets(player_match), on=GW_KEYS, how="left")
    player_gw["n_fixtures"] = player_gw["n_fixtures"].fillna(0).astype(int)
    player_gw = player_gw.merge(gameweeks, on=["season", "gw"], how="left")
    return add_rolling_features(player_gw)


def main() -> None:
    player_match = pd.read_parquet(PROCESSED / "player_match.parquet")
    gameweeks = pd.read_parquet(PROCESSED / "gameweeks.parquet")
    player_gw = build(player_match, gameweeks)
    player_gw.to_parquet(PROCESSED / "player_gw.parquet", index=False)
    print(f"player_gw {player_gw.shape}")


if __name__ == "__main__":
    main()
