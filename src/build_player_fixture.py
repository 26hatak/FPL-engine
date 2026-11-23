"""Build data/processed/player_fixture.parquet: one row per player per FIXTURE,
played and future, with features frozen at that gameweek's deadline.

Run from the project root: python src/build_player_fixture.py

Why per fixture and not per gameweek (DATA.md, "Double gameweeks"): a double
has two opponents, and E[points in the GW] = sum over its fixtures of
E[points | fixture]. A blank has no fixture rows, so it sums to 0 without a
special case.

Columns:
  keys         season, player_code, fixture, gw, kickoff_time, team_code,
               opponent_code, home, position
  fixture      n_fixtures_gw (his team's matches that GW), rest_days_before,
               rest_days_after (days since / until his team's previous / next match)
  f_*, snap_*  form and snapshot features of the GAMEWEEK, from build_player_gw:
               frozen at the GW deadline and shared by every fixture in that GW
  outcomes     minutes, goals_scored, ... for that one match (NaN for future)
  is_future    True for matches not played yet (the rows predictions are for)

Leakage rule: match 2 of a double must not see match 1, because the transfer
decision is made before both. That's why features are joined by GAMEWEEK from
player_gw (rolled per GW, shifted by one GW) and never rolled per match here.
"""
from pathlib import Path

import pandas as pd

from build_player_gw import GW_KEYS, add_rolling_features, add_snapshot_features
from build_player_match import CURRENT_SEASON, OUTCOMES

PROCESSED = Path("data/processed")
FIXTURE_KEYS = ["season", "player_code", "fixture"]


def team_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, fixture, team_code): both sides of every scheduled
    match, with opponent_code, home, gw, kickoff_time, finished, plus
      n_fixtures_gw     the team's matches in that gameweek (2 = double)
      rest_days_before  days since the team's previous match this season (NaN for the first)
      rest_days_after   days until its next match this season (NaN for the last)
    All known before the deadline: the schedule is public. Matches without a
    gameweek (postponed, not rescheduled yet) are left out."""
    fx = fixtures[fixtures["gw"].notna()]
    common = ["season", "fixture", "gw", "kickoff_time", "finished"]
    home = fx[common + ["team_h_code", "team_a_code"]].set_axis(
        common + ["team_code", "opponent_code"], axis=1).assign(home=True)
    away = fx[common + ["team_a_code", "team_h_code"]].set_axis(
        common + ["team_code", "opponent_code"], axis=1).assign(home=False)
    sides = pd.concat([home, away], ignore_index=True)
    sides["gw"] = sides["gw"].astype(int)
    sides = sides.sort_values(["season", "team_code", "kickoff_time"]).reset_index(drop=True)

    by_team = sides.groupby(["season", "team_code"])["kickoff_time"]
    day = pd.Timedelta(days=1)
    sides["rest_days_before"] = (sides["kickoff_time"] - by_team.shift(1)) / day
    sides["rest_days_after"] = (by_team.shift(-1) - sides["kickoff_time"]) / day
    sides["n_fixtures_gw"] = sides.groupby(["season", "team_code", "gw"])["fixture"].transform("size")
    return sides


def extend_to_future(player_gw: pd.DataFrame, roster: pd.DataFrame, gameweeks: pd.DataFrame,
                     season: str = CURRENT_SEASON) -> pd.DataFrame:
    """player_gw plus one empty row per roster player for every gameweek of
    `season` after the last one with data. f_* and snap_* are then recomputed
    by the SAME functions that built player_gw.

    An empty row (n_fixtures = 0, outcomes NaN) is exactly a blank to
    add_rolling_features: it carries the state at the end of the last played
    gameweek forward. So every future gameweek sees the form "as of now",
    frozen at the next deadline. Nothing is rolled per match.

    Only players with some history get rows (their club and position from the
    roster: today's club is the right one for future matches). A roster player
    with no history at all has no rows, so his features stay NaN.
    """
    in_season = player_gw[player_gw["season"] == season]
    last_gw = int(in_season["gw"].max()) if len(in_season) else 0
    future_gws = gameweeks.loc[(gameweeks["season"] == season) & (gameweeks["gw"] > last_gw), "gw"]
    known = roster[roster["player_code"].isin(player_gw["player_code"])]
    future = (known[["player_code", "team_code", "position"]]
              .merge(pd.DataFrame({"gw": future_gws.astype(int)}), how="cross")
              .assign(season=season, n_fixtures=0, n_home=0)
              .merge(gameweeks, on=["season", "gw"], how="left"))

    base = player_gw[[c for c in player_gw.columns
                      if not c.startswith(("f_", "snap_"))]]
    out = pd.concat([base, future], ignore_index=True)
    return add_rolling_features(out)


def build(player_match: pd.DataFrame, player_gw: pd.DataFrame, fixtures: pd.DataFrame,
          roster: pd.DataFrame, gameweeks: pd.DataFrame,
          players_daily: pd.DataFrame | None = None,
          season: str = CURRENT_SEASON) -> pd.DataFrame:
    """player_fixture: played matches from player_match, plus every roster
    player x every unplayed match of his current club this season.

    player_gw must come from build_player_gw.build (it supplies the GW grid
    and deadlines); its f_* and snap_* are recomputed here together with the
    future rows, so both always agree.
    """
    sides = team_fixtures(fixtures)
    gw_features = extend_to_future(player_gw, roster, gameweeks, season)
    if players_daily is not None:
        gw_features = add_snapshot_features(gw_features, players_daily)
    feature_cols = [c for c in gw_features.columns if c.startswith(("f_", "snap_"))]

    # Played: one row per player_match row (his club AT THE TIME of the match).
    played = player_match[["season", "player_code", "fixture", "gw", "kickoff_time",
                           "team_code", "opponent_code", "was_home", "position", *OUTCOMES]]
    # vaastav 2025-26 repeats 10 rows exactly (2 players); one match = one row.
    played = played.drop_duplicates(FIXTURE_KEYS)
    played = played.rename(columns={"was_home": "home"}).assign(is_future=False)
    played = played.merge(
        sides[["season", "fixture", "team_code", "n_fixtures_gw",
               "rest_days_before", "rest_days_after"]],
        on=["season", "fixture", "team_code"], how="left")

    # Future: roster player x his club's unplayed matches.
    upcoming = sides[(sides["season"] == season) & ~sides["finished"]]
    future = (roster[["player_code", "team_code", "position"]]
              .merge(upcoming.drop(columns="finished"), on="team_code")
              .assign(is_future=True))
    # A match counts as played once it is in player_match; never both.
    seen = played.loc[played["season"] == season, "fixture"].unique()
    future = future[~future["fixture"].isin(seen)]

    rows = pd.concat([played, future], ignore_index=True)
    rows = rows.merge(gw_features[GW_KEYS + feature_cols], on=GW_KEYS, how="left")
    rows["home"] = rows["home"].astype(bool)
    rows["team_code"] = rows["team_code"].astype(int)
    rows["opponent_code"] = rows["opponent_code"].astype(int)
    return rows.sort_values(["season", "gw", "kickoff_time", "player_code"]).reset_index(drop=True)


def main() -> None:
    player_match = pd.read_parquet(PROCESSED / "player_match.parquet")
    player_gw = pd.read_parquet(PROCESSED / "player_gw.parquet")
    fixtures = pd.read_parquet(PROCESSED / "fixtures.parquet")
    players = pd.read_parquet(PROCESSED / "players.parquet")
    gameweeks = pd.read_parquet(PROCESSED / "gameweeks.parquet")
    players_daily = pd.read_parquet(PROCESSED / "players_daily.parquet")
    roster = players[players["season"] == CURRENT_SEASON]

    pf = build(player_match, player_gw, fixtures, roster, gameweeks, players_daily)
    pf.to_parquet(PROCESSED / "player_fixture.parquet", index=False)
    n_future = int(pf["is_future"].sum())
    print(f"player_fixture {pf.shape}  ({n_future} future rows)")


if __name__ == "__main__":
    main()
