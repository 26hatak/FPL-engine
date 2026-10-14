"""Save raw FPL API snapshots.

Daily run (the GitHub job): python src/snapshot.py
Force a history refresh:    python src/snapshot.py --backfill

Saves gzipped JSON under data/raw/. Never edit these files; everything
downstream (features, models) is rebuilt from them.

Every run saves the daily snapshot and any newly finished gameweek. Player
histories (element_summary) are re-downloaded automatically once a gameweek
has finished that the newest history folder doesn't cover yet: about once a
week, ~12 minutes of requests.
"""
import datetime as dt
import gzip
import json
import sys
import time
from pathlib import Path

import requests

BASE = "https://fantasy.premierleague.com/api"
RAW = Path("data/raw")

session = requests.Session()
# Identify yourself politely; put your own contact here.
session.headers.update({"User-Agent": "fpl-engine student project (26hatak@gmail.com)"})


def get(path: str) -> dict:
    """GET one endpoint, retrying a few times with growing waits."""
    for attempt in range(3):
        r = session.get(f"{BASE}/{path}", timeout=30)
        if r.status_code == 200:
            return r.json()
        time.sleep(5 * (attempt + 1))
    r.raise_for_status()


def save(obj: dict, path: Path) -> None:
    """Write JSON gzipped (about 8x smaller, so a year of snapshots fits in git)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f)


def daily_snapshot() -> dict:
    """Prices, ownership, status flags, chance of playing, ep_next, plus fixtures."""
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H%MZ")
    folder = RAW / "daily" / stamp
    boot = get("bootstrap-static/")
    save(boot, folder / "bootstrap-static.json.gz")
    time.sleep(1)
    save(get("fixtures/"), folder / "fixtures.json.gz")
    return boot


def save_gameweek_stats(boot: dict) -> None:
    """One request per finished gameweek returns every player's stats for that week.
    Weeks already on disk are skipped, so this is safe to run every day."""
    finished = [e["id"] for e in boot["events"] if e["finished"] and e["data_checked"]]
    for gw in finished:
        out = RAW / "gameweeks" / f"gw{gw:02d}_live.json.gz"
        if out.exists():
            continue
        save(get(f"event/{gw}/live/"), out)
        time.sleep(1)


def histories_are_stale(boot: dict) -> bool:
    """True when a finished (and FPL-checked) gameweek is missing from the newest
    element_summary folder. Uses the same rule as save_gameweek_stats, so the
    history and the gwNN_live files always cover the same gameweeks."""
    finished = [e["id"] for e in boot["events"] if e["finished"] and e["data_checked"]]
    folders = sorted((RAW / "element_summary").glob("*"))
    if not finished:
        return False
    if not folders:
        return True
    rounds = [m["round"] for p in folders[-1].glob("*.json.gz")
              for m in json.load(gzip.open(p, "rt"))["history"]]
    return max(rounds, default=0) < max(finished)


def backfill_player_histories(boot: dict) -> None:
    """Each player's full history this season, including price ('value') and
    ownership ('selected') at every past gameweek, into a folder named by
    today's date. Files already there are skipped, so an interrupted run
    resumes the same day. Takes ~12 minutes."""
    folder = RAW / "element_summary" / dt.date.today().isoformat()
    for p in boot["elements"]:
        out = folder / f"{p['id']}.json.gz"
        if out.exists():
            continue
        save(get(f"element-summary/{p['id']}/"), out)
        time.sleep(1)


if __name__ == "__main__":
    boot = daily_snapshot()
    save_gameweek_stats(boot)
    if "--backfill" in sys.argv or histories_are_stale(boot):
        backfill_player_histories(boot)
    print(f"Saved snapshot: {len(boot['elements'])} players, {len(boot['events'])} gameweeks")
