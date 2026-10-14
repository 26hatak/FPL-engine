"""Build the reference tables: fixtures, teams, players (data/processed/).

Run from the project root: python src/build_fixtures.py

fixtures: one row per match, all seasons, PLAYED AND FUTURE. Future rows are
          what predictions are made for (opponent, home/away, kickoff); scores
          stay empty until the match is played. The daily snapshot refreshes the
          current season, so postponements and new kickoff times flow in on
          their own.
teams:    one row per team per season: season id, stable code, names.
players:  one row per player per season: stable code, names, club, position.

Models join these by code; they exist so any table can be given names or
future matches without copying them into every row.
"""
import pandas as pd

from build_player_match import CURRENT_SEASON, PAST_SEASONS, PROCESSED, RAW, load_json

POSITION = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}   # element_type; 5 = 2024-25 AM chip


def season_sources(season: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(fixtures, teams, players) as the source has them, for one season."""
    if season == CURRENT_SEASON:
        latest = sorted((RAW / "daily").iterdir())[-1]
        boot = load_json(latest / "bootstrap-static.json.gz")
        return (pd.DataFrame(load_json(latest / "fixtures.json.gz")),
                pd.DataFrame(boot["teams"]), pd.DataFrame(boot["elements"]))
    folder = RAW / "vaastav" / season
    return (pd.read_csv(folder / "fixtures.csv.gz"), pd.read_csv(folder / "teams.csv.gz"),
            pd.read_csv(folder / "players_raw.csv.gz"))


def build_season(season: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fx, tm, pl = season_sources(season)

    teams = pd.DataFrame({"season": season, "team_id": tm["id"], "team_code": tm["code"],
                          "team_name": tm["name"], "team_short": tm["short_name"]})
    code = teams.set_index("team_id")["team_code"]
    name = teams.set_index("team_id")["team_name"]

    fixtures = pd.DataFrame({
        "season": season,
        "fixture": fx["id"],
        "gw": pd.to_numeric(fx["event"]).astype("Int64"),   # empty while a match is unscheduled
        "kickoff_time": pd.to_datetime(fx["kickoff_time"], utc=True),
        "finished": fx["finished"].astype(bool),
        "team_h_code": fx["team_h"].map(code), "team_a_code": fx["team_a"].map(code),
        "team_h_name": fx["team_h"].map(name), "team_a_name": fx["team_a"].map(name),
        "team_h_score": pd.to_numeric(fx["team_h_score"]).astype("Int64"),
        "team_a_score": pd.to_numeric(fx["team_a_score"]).astype("Int64"),
        "team_h_difficulty": fx["team_h_difficulty"], "team_a_difficulty": fx["team_a_difficulty"],
    })

    pl = pl[pl["element_type"].isin(POSITION)]
    players = pd.DataFrame({
        "season": season, "player_code": pl["code"], "element": pl["id"],
        "web_name": pl["web_name"], "first_name": pl["first_name"],
        "second_name": pl["second_name"], "team_code": pl["team"].map(code),
        "position": pl["element_type"].map(POSITION),
    })
    return fixtures, teams, players


def main() -> None:
    parts = [build_season(s) for s in PAST_SEASONS + [CURRENT_SEASON]]
    names = ["fixtures", "teams", "players"]
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        table = pd.concat([p[i] for p in parts], ignore_index=True)
        table.to_parquet(PROCESSED / f"{name}.parquet", index=False)
        print(f"{name} {table.shape}", end="  ")
    print()


if __name__ == "__main__":
    main()
