"""Build data/processed/xp.parquet: v0 expected points for the next 6 gameweeks.

Run from the project root: python src/predict_baseline.py

v0 is the crudest prediction that lets the wildcard optimizer run end to end
(gate Nov 8). Phase 1 replaces it with the decomposed model; it then stays as
the baseline that model must beat.

    xp = pts_per_match  x  n_fixtures  x  availability
         (player form)     (schedule)     (injury news)

xp: one row per (player_code, gw) for every player in the game now:
  season, player_code, gw, team_code, position,
  pts_per_match   total points / matches over his last 5 gameweeks with a match
  n_matches       matches behind that rate (0 = no history, so xp = 0)
  n_fixtures      his team's matches that gameweek (0 blank, 2 double)
  availability    chance of playing / 100 for the NEXT gameweek only; 0 for
                  every gameweek if he has left the club (status "u"); else 1
  xp              the prediction
  as_of           the snapshot it was made from

Leakage rule: a prediction for gameweeks starting at first_gw only uses
player_gw rows from before first_gw, and the snapshot is from before the
first_gw deadline. tests/test_predict_baseline.py checks both.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from build_player_match import CURRENT_SEASON

PROCESSED = Path("data/processed")
HORIZON = 6   # gameweeks ahead, as the optimizer plans
WINDOW = 5    # gameweeks of form, the "last 5" baseline from DECISIONS.md


def points_per_match(player_gw: pd.DataFrame, season: str, first_gw: int,
                     window: int = WINDOW) -> pd.DataFrame:
    """Per player_code: pts_per_match and n_matches over his last `window`
    gameweeks with a match, using ONLY gameweeks before (season, first_gw).

    Per match, not per gameweek: a double is 2 matches, so dividing by
    gameweeks would inflate a player who just had one.
    "Gameweeks with a match" counts 0-minute matches too, so a benched or
    injured player's rate drops. That is the only minutes signal v0 has.
    The window crosses seasons, like the f_* features.
    """
    before = (player_gw["season"] < season) | (
        (player_gw["season"] == season) & (player_gw["gw"] < first_gw))
    played = player_gw[before & (player_gw["n_fixtures"] > 0)]
    last = played.sort_values(["season", "gw"]).groupby("player_code").tail(window)
    agg = last.groupby("player_code").agg(points=("total_points", "sum"),
                                          n_matches=("n_fixtures", "sum"))
    return pd.DataFrame({"pts_per_match": agg["points"] / agg["n_matches"],
                         "n_matches": agg["n_matches"]})


def team_fixture_counts(fixtures: pd.DataFrame, season: str, gws: list[int]) -> pd.DataFrame:
    """Matches per (team_code, gw) for the given gameweeks, every team in the
    season included, 0 = blank. Unscheduled matches (gw empty) count nowhere."""
    fx = fixtures[(fixtures["season"] == season) & fixtures["gw"].isin(gws)]
    sides = pd.concat([fx[["gw", "team_h_code"]].set_axis(["gw", "team_code"], axis=1),
                       fx[["gw", "team_a_code"]].set_axis(["gw", "team_code"], axis=1)])
    counts = sides.groupby(["team_code", "gw"]).size()

    season_fx = fixtures[fixtures["season"] == season]
    teams = pd.unique(season_fx[["team_h_code", "team_a_code"]].values.ravel())
    grid = pd.MultiIndex.from_product([sorted(teams), gws], names=["team_code", "gw"])
    return counts.reindex(grid, fill_value=0).rename("n_fixtures").reset_index()


def predict(player_gw: pd.DataFrame, fixtures: pd.DataFrame, roster: pd.DataFrame,
            season: str, first_gw: int, horizon: int = HORIZON,
            snapshot: pd.DataFrame | None = None) -> pd.DataFrame:
    """xp for every roster player (player_code, team_code, position) for
    gameweeks first_gw .. first_gw + horizon - 1, cut at the season's end.

    snapshot: one players_daily row per player, taken before the first_gw
    deadline. Without it (past seasons have none) availability is 1.
    """
    last_gw = int(fixtures.loc[fixtures["season"] == season, "gw"].max())
    gws = list(range(first_gw, min(first_gw + horizon - 1, last_gw) + 1))

    xp = (roster[["player_code", "team_code", "position"]]
          .merge(pd.DataFrame({"gw": gws}), how="cross")
          .merge(team_fixture_counts(fixtures, season, gws), on=["team_code", "gw"], how="left")
          .merge(points_per_match(player_gw, season, first_gw), on="player_code", how="left"))
    xp["n_fixtures"] = xp["n_fixtures"].fillna(0).astype(int)   # a team not in the schedule: no match
    # No history (a new signing): v0 can't rate him, so 0. Phase 1 shrinks to a position prior instead.
    xp["n_matches"] = xp["n_matches"].fillna(0).astype(int)
    xp["pts_per_match"] = xp["pts_per_match"].fillna(0.0)

    xp["availability"] = 1.0
    if snapshot is not None:
        snap = snapshot.set_index("player_code")
        # chance_of_playing_next_round is about the next gameweek only; empty = no news = fit.
        chance = xp["player_code"].map(snap["chance_of_playing_next_round"] / 100).fillna(1.0)
        xp["availability"] = np.where(xp["gw"] == first_gw, chance, 1.0)
        # "u" = left the club or loaned out: out for the whole horizon, not just next week.
        gone = xp["player_code"].map(snap["status"]).eq("u")
        xp.loc[gone, "availability"] = 0.0

    xp["xp"] = xp["pts_per_match"] * xp["n_fixtures"] * xp["availability"]
    xp.insert(0, "season", season)
    cols = ["season", "player_code", "gw", "team_code", "position", "pts_per_match",
            "n_matches", "n_fixtures", "availability", "xp"]
    return xp[cols].sort_values(["gw", "player_code"]).reset_index(drop=True)


def main() -> None:
    player_gw = pd.read_parquet(PROCESSED / "player_gw.parquet")
    fixtures = pd.read_parquet(PROCESSED / "fixtures.parquet")
    players = pd.read_parquet(PROCESSED / "players.parquet")
    gameweeks = pd.read_parquet(PROCESSED / "gameweeks.parquet")
    players_daily = pd.read_parquet(PROCESSED / "players_daily.parquet")

    # "Now" = the latest snapshot, not the wall clock, so a rebuild from the
    # same raw data always gives the same table.
    as_of = players_daily["snapshot_time"].max()
    snapshot = players_daily[players_daily["snapshot_time"] == as_of]
    upcoming = gameweeks[(gameweeks["season"] == CURRENT_SEASON) & (gameweeks["deadline_time"] > as_of)]
    if upcoming.empty:
        print("xp: no gameweek deadline left this season, nothing to predict")
        return
    first_gw = int(upcoming["gw"].min())

    roster = players[players["season"] == CURRENT_SEASON]
    xp = predict(player_gw, fixtures, roster, CURRENT_SEASON, first_gw, snapshot=snapshot)
    xp["as_of"] = as_of
    xp.to_parquet(PROCESSED / "xp.parquet", index=False)
    print(f"xp {xp.shape}  GW{xp['gw'].min()}-{xp['gw'].max()}  as of {as_of:%Y-%m-%d %H:%M} UTC")


if __name__ == "__main__":
    main()
