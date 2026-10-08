"""Build data/processed/player_match.parquet and gameweeks.parquet from data/raw.

Run from the project root: python src/build_player_match.py

player_match: one row per player per MATCH, all seasons, same columns.
gameweeks:    one row per season per gameweek, with its deadline (the leakage clock).

Two functions are left for you (marked YOUR TURN). Their tests are in
tests/test_player_match.py: run `pytest tests/test_player_match.py -q`.
"""
import gzip
import json
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
CURRENT_SEASON = "2026-27"
PAST_SEASONS = ["2024-25", "2025-26"]

KEYS = ["season", "player_code", "element", "fixture", "gw", "kickoff_time",
        "team_code", "opponent_code", "was_home", "position"]
OUTCOMES = ["minutes", "starts", "goals_scored", "assists", "clean_sheets",
            "goals_conceded", "saves", "bonus", "bps", "expected_goals",
            "expected_assists", "expected_goals_conceded", "defensive_contribution",
            "total_points"]
# Known before the deadline of that gameweek (price, ownership, transfers).
PRE_DEADLINE = ["value", "selected", "transfers_in", "transfers_out"]
COLUMNS = KEYS + OUTCOMES + PRE_DEADLINE


def load_json(path: Path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """The API stores xG, xA and ICT as strings like "0.45". Cast every
    outcome and pre-deadline column to a number."""
    df = df.copy()
    for col in OUTCOMES + PRE_DEADLINE:
        if col in df:
            df[col] = pd.to_numeric(df[col])
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
    return df


# ---------------------------------------------------------------- YOUR TURN 1
def add_own_team(matches: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Add a 'team' column: the player's club IN THAT MATCH (season team id).

    Why: element_summary history only has 'opponent_team' and 'was_home'.
    players.team is today's club, which is wrong for anyone transferred
    mid-season, so it would silently corrupt team-strength and xG-share models.

    matches:  has 'fixture' (fixture id) and 'was_home' (bool)
    fixtures: has 'id', 'team_h', 'team_a'
    Returns matches with a new int column 'team', same rows, same order.

    Hint: merge on fixture id, then pick team_h or team_a with was_home
    (np.where or Series.where). Check the row count didn't change.
    """
    raise NotImplementedError("add_own_team: see docstring")


# ---------------------------------------------------------------- YOUR TURN 2
def combine_seasons(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack per-season frames into one table with exactly COLUMNS, in order.

    Seasons don't share all columns: defensive_contribution only exists from
    2025-26 (DefCon points started then). For older seasons it must be NaN,
    NOT 0: 0 would claim "made no defensive actions", NaN says "not recorded".

    Returns one DataFrame, columns == COLUMNS, sorted by
    season, kickoff_time, player_code, with a fresh 0..n-1 index.

    Hint: DataFrame.reindex(columns=...) adds missing columns as NaN.
    """
    raise NotImplementedError("combine_seasons: see docstring")


def load_current_season() -> pd.DataFrame:
    """This season from the API snapshots: element_summary history (per match)
    plus the latest bootstrap for stable codes and positions."""
    latest = sorted((RAW / "daily").iterdir())[-1]
    boot = load_json(latest / "bootstrap-static.json.gz")
    fixtures = pd.DataFrame(load_json(latest / "fixtures.json.gz"))
    summary_dir = sorted((RAW / "element_summary").iterdir())[-1]
    matches = pd.DataFrame(
        [m for p in summary_dir.glob("*.json.gz") for m in load_json(p)["history"]]
    )

    players = pd.DataFrame(boot["elements"]).set_index("id")
    team_code = pd.DataFrame(boot["teams"]).set_index("id")["code"]
    position = {t["id"]: t["singular_name_short"].replace("GKP", "GK")
                for t in boot["element_types"]}

    matches = add_own_team(matches, fixtures)
    matches["season"] = CURRENT_SEASON
    matches["gw"] = matches["round"]
    matches["player_code"] = matches["element"].map(players["code"])
    matches["position"] = matches["element"].map(players["element_type"]).map(position)
    matches["team_code"] = matches["team"].map(team_code)
    matches["opponent_code"] = matches["opponent_team"].map(team_code)
    return to_numeric(matches)


def load_past_season(season: str) -> pd.DataFrame:
    """A past season from vaastav: merged_gw (per match) plus players_raw and
    teams for stable codes. merged_gw already has the club at the time, by name."""
    folder = RAW / "vaastav" / season
    matches = pd.read_csv(folder / "merged_gw.csv.gz")
    players = pd.read_csv(folder / "players_raw.csv.gz").set_index("id")
    teams = pd.read_csv(folder / "teams.csv.gz")

    # 2024-25 lists Assistant Manager chip "players" as position AM.
    matches = matches[matches["position"] != "AM"].copy()
    matches["season"] = season
    matches["gw"] = matches["GW"]
    matches["player_code"] = matches["element"].map(players["code"])
    matches["team_code"] = matches["team"].map(teams.set_index("name")["code"])
    matches["opponent_code"] = matches["opponent_team"].map(teams.set_index("id")["code"])
    return to_numeric(matches)


def current_deadlines() -> pd.DataFrame:
    """This season's real deadlines from the latest bootstrap."""
    latest = sorted((RAW / "daily").iterdir())[-1]
    events = pd.DataFrame(load_json(latest / "bootstrap-static.json.gz")["events"])
    return pd.DataFrame({
        "season": CURRENT_SEASON,
        "gw": events["id"],
        "deadline_time": pd.to_datetime(events["deadline_time"], utc=True),
        "deadline_is_approx": False,
    })


def approx_deadlines(fixtures: pd.DataFrame, season: str) -> pd.DataFrame:
    """Past seasons have no deadline file. FPL's deadline is 90 minutes before
    the gameweek's first kickoff (exact for every 2026-27 GW so far; see
    tests/test_real_data.py). Caveat: vaastav's fixtures are end-of-season, so a
    postponed opening match makes the approximation too late for that GW."""
    first_kickoff = (pd.to_datetime(fixtures["kickoff_time"], utc=True)
                     .groupby(fixtures["event"]).min())
    return pd.DataFrame({
        "season": season,
        "gw": first_kickoff.index.astype(int),
        "deadline_time": (first_kickoff - pd.Timedelta(minutes=90)).to_numpy(),
        "deadline_is_approx": True,
    })


def main() -> None:
    frames = [load_past_season(s) for s in PAST_SEASONS] + [load_current_season()]
    player_match = combine_seasons(frames)

    gameweeks = pd.concat(
        [approx_deadlines(pd.read_csv(RAW / "vaastav" / s / "fixtures.csv.gz"), s)
         for s in PAST_SEASONS] + [current_deadlines()],
        ignore_index=True,
    )

    PROCESSED.mkdir(parents=True, exist_ok=True)
    player_match.to_parquet(PROCESSED / "player_match.parquet", index=False)
    gameweeks.to_parquet(PROCESSED / "gameweeks.parquet", index=False)
    print(f"player_match {player_match.shape}, gameweeks {gameweeks.shape}")


if __name__ == "__main__":
    main()
