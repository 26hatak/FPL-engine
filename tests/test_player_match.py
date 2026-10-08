import numpy as np
import pandas as pd

from build_player_match import COLUMNS, add_own_team, combine_seasons


def test_add_own_team_picks_home_or_away():
    fixtures = pd.DataFrame({"id": [10, 20], "team_h": [1, 3], "team_a": [2, 4]})
    matches = pd.DataFrame({"fixture": [20, 10, 10], "was_home": [False, True, False],
                            "element": [7, 8, 9]})
    out = add_own_team(matches, fixtures)
    assert out["team"].tolist() == [4, 1, 2]
    assert out["element"].tolist() == [7, 8, 9]   # same rows, same order


def test_combine_seasons_columns_and_defcon_nan(player_match):
    old = player_match[player_match["season"] == "2024-25"].drop(columns="defensive_contribution")
    old = old.assign(mng_win=1)                    # a column we don't want to keep
    new = player_match[player_match["season"] == "2025-26"]
    out = combine_seasons([new, old])

    assert list(out.columns) == COLUMNS
    assert len(out) == len(player_match)
    assert out.index.tolist() == list(range(len(out)))
    assert out["season"].is_monotonic_increasing
    # Not recorded -> NaN, never 0.
    assert out.loc[out["season"] == "2024-25", "defensive_contribution"].isna().all()
    assert np.isfinite(out.loc[out["season"] == "2025-26", "defensive_contribution"]).all()
