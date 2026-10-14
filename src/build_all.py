"""Rebuild every processed table from data/raw, in dependency order.

Run from the project root: python src/build_all.py

data/processed/ is never edited or committed: this script recreates it, so
after `git pull` brings new raw data, one run brings every table up to date.
The daily GitHub job runs it too (then pytest) as a health check.
"""
import build_fixtures
import build_player_gw
import build_player_match
import build_snapshots

if __name__ == "__main__":
    build_player_match.main()   # raw -> player_match, gameweeks
    build_fixtures.main()       # raw -> fixtures (played + future), teams, players
    build_snapshots.main()      # raw daily snapshots -> players_daily
    build_player_gw.main()      # all three -> player_gw
