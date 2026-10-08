"""Small synthetic data shared by the tests. Built by hand so every expected
number in the tests can be checked with a calculator."""
import pandas as pd
import pytest

from build_player_match import COLUMNS


def match(season, player, gw, fixture, minutes=90, **stats):
    """One player_match row; every stat not given is 0."""
    row = dict.fromkeys(COLUMNS, 0)
    row.update(season=season, player_code=player, element=player, gw=gw,
               fixture=fixture, team_code=1, opponent_code=2, was_home=True,
               position="MID", minutes=minutes, starts=int(minutes >= 60),
               kickoff_time=pd.Timestamp("2025-08-01", tz="UTC") + pd.Timedelta(days=7 * gw + fixture % 2))
    row.update(stats)
    return row


@pytest.fixture
def player_match():
    rows = [
        # 2024-25: player 100 only; DefCon not recorded that season.
        match("2024-25", 100, 1, 11, defensive_contribution=float("nan"), expected_goals=0.2),
        match("2024-25", 100, 2, 21, defensive_contribution=float("nan"), expected_goals=0.4),
        # 2025-26: player 100 plays every GW, two matches (a double) in GW3.
        match("2025-26", 100, 1, 11, goals_scored=1, expected_goals=0.5, defensive_contribution=4),
        match("2025-26", 100, 2, 21, expected_goals=0.1, defensive_contribution=6),
        match("2025-26", 100, 3, 31, goals_scored=1, expected_goals=0.7, defensive_contribution=5),
        match("2025-26", 100, 3, 32, goals_scored=2, expected_goals=0.9, defensive_contribution=3),
        match("2025-26", 100, 4, 41, expected_goals=0.3, defensive_contribution=8),
        match("2025-26", 100, 5, 51, expected_goals=0.2, defensive_contribution=2),
        match("2025-26", 100, 6, 61, expected_goals=0.6, defensive_contribution=7),
        # Player 200: blank in GW4 (no row), distinct minutes to check rolling means.
        match("2025-26", 200, 1, 12, minutes=10),
        match("2025-26", 200, 2, 22, minutes=20),
        match("2025-26", 200, 3, 33, minutes=30),
        match("2025-26", 200, 5, 52, minutes=50),
        match("2025-26", 200, 6, 62, minutes=60),
        # Player 300: first appears in GW3.
        match("2025-26", 300, 3, 34),
        match("2025-26", 300, 4, 42),
        match("2025-26", 300, 5, 53),
        match("2025-26", 300, 6, 63),
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


@pytest.fixture
def gameweeks():
    rows = [{"season": s, "gw": gw,
             "deadline_time": pd.Timestamp("2025-08-01", tz="UTC") + pd.Timedelta(days=7 * gw),
             "deadline_is_approx": False}
            for s, n in [("2024-25", 2), ("2025-26", 6)] for gw in range(1, n + 1)]
    return pd.DataFrame(rows)
