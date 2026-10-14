"""Build data/processed/players_daily.parquet from every daily snapshot.

Run from the project root: python src/build_snapshots.py

players_daily: one row per player per snapshot, i.e. what was KNOWN at that
moment: availability flags, price, ownership, FPL's own prediction. The daily
GitHub job adds a snapshot every day, so this table grows on its own; a rebuild
just reads all of data/raw/daily/ again.

build_player_gw.py joins it onto player_gw "as of" each deadline: the latest
snapshot strictly before the deadline, never one after.
"""
import datetime as dt
from pathlib import Path

import pandas as pd

from build_player_match import CURRENT_SEASON, RAW, PROCESSED, load_json

# Fields from bootstrap-static elements, as they stood at snapshot time.
FIELDS = ["status", "chance_of_playing_next_round", "news", "now_cost",
          "selected_by_percent", "ep_next", "form",
          "transfers_in_event", "transfers_out_event"]
# The API sends these as strings ("43.7"); null chance_of_playing means "no news".
NUMERIC = ["chance_of_playing_next_round", "now_cost", "selected_by_percent",
           "ep_next", "form", "transfers_in_event", "transfers_out_event"]


def snapshot_time(folder: Path) -> pd.Timestamp:
    """Folder names are the UTC time the snapshot was taken: 2026-10-14T2139Z."""
    return pd.Timestamp(dt.datetime.strptime(folder.name, "%Y-%m-%dT%H%MZ"), tz="UTC")


def load_snapshot(folder: Path) -> pd.DataFrame:
    elements = load_json(folder / "bootstrap-static.json.gz")["elements"]
    df = pd.DataFrame(elements)[["id", "code"] + FIELDS]
    df = df.rename(columns={"id": "element", "code": "player_code"})
    df.insert(0, "snapshot_time", snapshot_time(folder))
    df.insert(0, "season", CURRENT_SEASON)
    return df


def build_players_daily() -> pd.DataFrame:
    folders = sorted((RAW / "daily").iterdir())
    daily = pd.concat([load_snapshot(f) for f in folders], ignore_index=True)
    for col in NUMERIC:
        daily[col] = pd.to_numeric(daily[col])
    return daily.sort_values(["snapshot_time", "player_code"]).reset_index(drop=True)


def main() -> None:
    daily = build_players_daily()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(PROCESSED / "players_daily.parquet", index=False)
    print(f"players_daily {daily.shape}, {daily['snapshot_time'].nunique()} snapshots")


if __name__ == "__main__":
    main()
