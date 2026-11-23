"""Build data/processed/xp_components.parquet: v1 expected points, decomposed.

Run from the project root: python src/predict_components.py

Instead of predicting points directly, predict the EVENTS that score points
and multiply by the points per event from scoring.yaml (CLAUDE.md, Phase 1):

    xp = xp_appear + xp_goals + xp_assists + xp_cs + xp_saves
         + xp_conceded + xp_defcon + xp_bonus + xp_cards

Why: each piece has its own driver that a person can read and argue with
(p60 = 0.4 is a rotation risk; p_cs = 0.15 is a tough fixture), a rule change
is a YAML edit, and later ML only replaces how a driver is estimated, never
the sum. v1 has no ML: every driver is a rate or an average.

Drivers, per FIXTURE (a double has two rows, summed to the gameweek):
  p_play    P(minutes > 0)  = availability x share of his last 5 matches with minutes
  p60       P(60+ minutes)  = availability x share of his last 5 matches with 60+
  exp_mins  E[minutes]      = availability x minutes per match over his last 5
  lambda_g  expected goals per 90 in THIS fixture: his xG/90 over the last 10
            gameweeks, shrunk toward his position's average by sample size
            (empirical Bayes), x how leaky this opponent's defence is
  lambda_a  the same with xA
  p_cs      P(his team concedes 0) = exp(-lambda_against), lambda_against =
            opponent attack x own defence / league average x home factor
  p_defcon  P(reaching the DefCon threshold | plays 60+), Poisson on his
            shrunk DefCon/90
  opponent, home

Leakage rule: everything for gameweek t uses only gameweeks before t: the f_*
features (player_fixture), and every estimated constant here (team ratings,
league average, home factor, position priors, bonus coefficients) is frozen at
the GW deadline in the same way: summed per gameweek, accumulated, shifted by
one gameweek. tests/test_components.py perturbs the future to check.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from build_player_match import CURRENT_SEASON

PROCESSED = Path("data/processed")
SCORING = Path("scoring.yaml")
HORIZON = 6
MIN_WINDOW = 5        # minutes drivers: last 5 gameweeks with a match, like the v0 baseline
RATE_WINDOW = 10      # per-90 rates: longer, since shrinkage takes care of small samples
# How many full matches of evidence the position prior is worth, per stat and
# position (m in `shrink`). Estimated once by split-half reliability on 2024-25
# (odd vs even matches, players with 450+ minutes in each half):
# m = 90s per half x (1 - rho) / rho. A defender's xG is mostly noise (set
# pieces), so his own rate needs ~28 matches to outweigh the prior; DefCon is
# very stable (~3). GK xG / xA: too few to estimate, use the defender value.
# DECISIONS.md 2026-11-22 has the table.
PRIOR_90S = {
    "xg": {"GK": 28, "DEF": 28, "MID": 4, "FWD": 13},
    "xa": {"GK": 3, "DEF": 3, "MID": 5, "FWD": 8},
    "saves": {"GK": 18, "DEF": 18, "MID": 18, "FWD": 18},     # only used for GK
    "defcon": {"GK": 3, "DEF": 3, "MID": 3, "FWD": 6},         # 2025-26 (first DefCon season)
}
TEAM_WINDOW = 10      # team xG for / against: last 10 matches
TEAM_PRIOR_MATCHES = 5  # team ratings: shrunk toward the league average by 5 matches
POISSON_KMAX = 60

GW = ["season", "gw"]
COMPONENTS = ["xp_appear", "xp_goals", "xp_assists", "xp_cs", "xp_saves",
              "xp_conceded", "xp_defcon", "xp_bonus", "xp_cards"]
DRIVERS = ["p_play", "p60", "exp_mins", "lambda_g", "lambda_a", "p_cs", "p_defcon",
           "opponent", "home"]
BONUS_X = ["b_60", "b_sub", "b_goal", "b_assist", "b_cs"]


def load_scoring(path: Path = SCORING) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ helpers

def gw_order(gameweeks: pd.DataFrame) -> pd.DataFrame:
    """Every (season, gw) in time order, with t = 0, 1, 2, ... and the
    previous gameweek's deadline (NaT for a season's first gameweek)."""
    order = gameweeks[GW + ["deadline_time"]].sort_values(GW).reset_index(drop=True)
    order["t"] = np.arange(len(order))
    order["prev_deadline"] = order.groupby("season")["deadline_time"].shift(1)
    return order


def frozen_cumsum(per_gw: pd.DataFrame, by: str, order: pd.DataFrame) -> pd.DataFrame:
    """per_gw: sums per (season, gw, by). Returns, for EVERY (season, gw, by)
    in `order`, the total over all gameweeks strictly before it.

    The deadline rule for any estimated constant: add up each gameweek, run a
    cumulative sum in time order, shift by one gameweek, so gameweek t only
    sees gameweeks < t. Gameweeks without data add 0 (future ones carry the
    last state)."""
    values = [c for c in per_gw.columns if c not in GW + [by]]
    grid = order[GW + ["t"]].merge(pd.DataFrame({by: per_gw[by].unique()}), how="cross")
    out = grid.merge(per_gw, on=GW + [by], how="left").sort_values([by, "t"])
    out[values] = out[values].fillna(0.0)
    out[values] = out.groupby(by)[values].cumsum()
    out[values] = out.groupby(by)[values].shift(1).fillna(0.0)
    return out.drop(columns="t").reset_index(drop=True)


def poisson_pmf(lam, kmax: int = POISSON_KMAX) -> np.ndarray:
    """P(X = k) for k = 0..kmax, one row per lambda (X ~ Poisson(lambda)).
    By recursion p_k = p_{k-1} * lambda / k: no factorials, lambda = 0 works."""
    lam = np.asarray(lam, dtype=float)
    pmf = np.empty((len(lam), kmax + 1))
    pmf[:, 0] = np.exp(-lam)
    for k in range(1, kmax + 1):
        pmf[:, k] = pmf[:, k - 1] * lam / k
    return pmf


def expected_floor_div(lam, per: int) -> np.ndarray:
    """E[floor(X / per)], X ~ Poisson(lambda): '1 point per 3 saves' is not
    lambda / 3 (2 saves score nothing), so sum over the distribution."""
    k = np.arange(POISSON_KMAX + 1)
    return poisson_pmf(lam) @ (k // per)


def prob_at_least(lam, threshold) -> np.ndarray:
    """P(X >= threshold), X ~ Poisson(lambda); threshold may differ per row."""
    k = np.arange(POISSON_KMAX + 1)
    below = k[None, :] < np.asarray(threshold, dtype=float)[:, None]
    return 1.0 - (poisson_pmf(lam) * below).sum(axis=1)


def shrink(rate90, minutes, prior90, prior_90s):
    """Empirical-Bayes posterior mean of a per-90 rate (Gamma-Poisson):

        (events + m * prior) / (90s played + m)

    m = prior_90s (a number, or one per row). With 0 minutes the answer is
    the prior; with many more 90s than m it is mostly the player's own rate.
    A 1-match player with 2 xG is pulled most of the way back to his
    position's average: one game says little, and the optimizer would
    otherwise buy every one-week wonder."""
    n90 = pd.Series(minutes).fillna(0.0) / 90
    events = (pd.Series(rate90) * n90).fillna(0.0)
    return (events + prior_90s * prior90) / (n90 + prior_90s)


# ------------------------------------------------------------- estimated constants

def team_strength(pf: pd.DataFrame, gameweeks: pd.DataFrame, window: int = TEAM_WINDOW,
                  prior_matches: float = TEAM_PRIOR_MATCHES) -> pd.DataFrame:
    """Per (season, gw, team_code), frozen at the GW deadline:
      att        xG per match over the team's last `window` matches, shrunk
                 toward the league average by `prior_matches` matches
      dfn        the same for xG conceded (higher = leakier)
      league_xg  league average xG per team per match, all earlier gameweeks
      home_adv   sqrt(home xG / away xG), all earlier gameweeks: a home team
                 gets x home_adv, an away team / home_adv

    Team xG = sum of its players' expected_goals in the match (from the
    played rows of player_fixture). Promoted teams start at the league average.
    """
    played = pf[~pf["is_future"]]
    team = (played.groupby(["season", "fixture", "gw", "kickoff_time", "team_code",
                            "opponent_code", "home"], as_index=False)["expected_goals"]
            .sum(min_count=1).rename(columns={"expected_goals": "xg_for"}))
    against = team[["season", "fixture", "team_code", "xg_for"]].rename(
        columns={"team_code": "opponent_code", "xg_for": "xg_against"})
    team = team.merge(against, on=["season", "fixture", "opponent_code"], how="left")
    order = gw_order(gameweeks)

    # League average and home advantage: cumulative over all earlier gameweeks.
    team["xg_home"] = team["xg_for"].where(team["home"])
    team["xg_away"] = team["xg_for"].where(~team["home"])
    league = team.groupby(GW).agg(xg=("xg_for", "sum"), n=("xg_for", "count"),
                                  xg_h=("xg_home", "sum"), n_h=("xg_home", "count"),
                                  xg_a=("xg_away", "sum"), n_a=("xg_away", "count"))
    league = frozen_cumsum(league.reset_index().assign(all="all"), "all", order)
    league["league_xg"] = league["xg"] / league["n"].where(league["n"] > 0)
    league["home_adv"] = np.sqrt((league["xg_h"] / league["n_h"]) /
                                 (league["xg_a"] / league["n_a"]))
    league = league[GW + ["league_xg", "home_adv"]]

    # Team form: rolling over its last `window` matches, end-of-GW state, then
    # forward-filled over gameweeks without a match and shifted one gameweek.
    team = team.sort_values(["team_code", "kickoff_time"])
    by_team = team.groupby("team_code")
    for col in ["xg_for", "xg_against"]:
        roll = by_team[col].rolling(window, min_periods=1)
        team[f"{col}_sum"] = roll.sum().reset_index(level=0, drop=True)
        team[f"{col}_n"] = roll.count().reset_index(level=0, drop=True)
    state_cols = ["xg_for_sum", "xg_for_n", "xg_against_sum", "xg_against_n"]
    state = team.groupby(GW + ["team_code"])[state_cols].last().reset_index()

    teams = pd.concat([pf[["season", "team_code"]],
                       pf[["season", "opponent_code"]].set_axis(["season", "team_code"], axis=1)]
                      ).drop_duplicates()
    grid = order[GW + ["t"]].merge(teams, on="season").merge(state, on=GW + ["team_code"], how="left")
    grid = grid.sort_values(["team_code", "t"])
    grid[state_cols] = grid.groupby("team_code")[state_cols].ffill()
    grid[state_cols] = grid.groupby("team_code")[state_cols].shift(1)

    out = grid.merge(league, on=GW, how="left")
    k, avg = prior_matches, out["league_xg"]
    out["att"] = (out["xg_for_sum"].fillna(0) + k * avg) / (out["xg_for_n"].fillna(0) + k)
    out["dfn"] = (out["xg_against_sum"].fillna(0) + k * avg) / (out["xg_against_n"].fillna(0) + k)
    return out[GW + ["team_code", "att", "dfn", "league_xg", "home_adv"]].reset_index(drop=True)


def position_priors(pf: pd.DataFrame, gameweeks: pd.DataFrame, scoring: dict) -> pd.DataFrame:
    """Per (season, gw, position), frozen at the GW deadline, over ALL earlier
    played matches (minutes-weighted):
      prior_xg90, prior_xa90, prior_saves90, prior_defcon90 (DefCon-recorded
      minutes only), card_pts90 (card points per 90 from scoring.yaml)
      b_60, b_sub, b_goal, b_assist, b_cs: bonus = least squares of bonus on
      [played 60+, played 1-59, goals, assists, clean sheet], no intercept
      (no minutes, no bonus). Solved from accumulated X'X and X'y, so it is
      also frozen at the deadline.
    """
    played = pf[~pf["is_future"] & (pf["minutes"] > 0)].copy()
    cards = scoring["cards"]
    played["card_pts"] = cards["yellow"] * played["yellow_cards"] + cards["red"] * played["red_cards"]
    played["defcon_minutes"] = played["minutes"].where(played["defensive_contribution"].notna())

    X = np.column_stack([played["minutes"] >= 60, played["minutes"] < 60,
                         played["goals_scored"], played["assists"],
                         played["clean_sheets"]]).astype(float)
    y = played["bonus"].to_numpy(float)
    p = X.shape[1]
    xtx = pd.DataFrame((X[:, :, None] * X[:, None, :]).reshape(len(X), -1),
                       columns=[f"xtx{i}" for i in range(p * p)], index=played.index)
    xty = pd.DataFrame(X * y[:, None], columns=[f"xty{i}" for i in range(p)], index=played.index)
    played = played.join(xtx).join(xty)

    sums = ["minutes", "expected_goals", "expected_assists", "saves", "card_pts",
            "defensive_contribution", "defcon_minutes", *xtx.columns, *xty.columns]
    per_gw = played.groupby(GW + ["position"])[sums].sum().reset_index()
    cum = frozen_cumsum(per_gw, "position", gw_order(gameweeks))

    def per90(col, minutes="minutes"):
        return cum[col] / cum[minutes].where(cum[minutes] > 0) * 90

    out = cum[GW + ["position"]].copy()
    out["prior_xg90"] = per90("expected_goals")
    out["prior_xa90"] = per90("expected_assists")
    out["prior_saves90"] = per90("saves")
    out["prior_defcon90"] = per90("defensive_contribution", "defcon_minutes")
    out["card_pts90"] = per90("card_pts")

    A = cum[xtx.columns].to_numpy().reshape(-1, p, p)
    b = cum[xty.columns].to_numpy()
    betas = np.array([np.linalg.lstsq(a, v, rcond=None)[0] for a, v in zip(A, b)])
    out[BONUS_X] = betas
    return out


# ------------------------------------------------------------------ the model

def availability(rows: pd.DataFrame) -> pd.Series:
    """chance_of_playing_next_round / 100 where the snapshot is about THIS
    gameweek (taken after the previous deadline), 1 otherwise or with no news;
    0 for every gameweek if status is "u" (left the club or on loan).

    Same rule as predict_baseline, said without "now": a flag only speaks
    about the next round, so it applies to the row's gameweek only if no
    other deadline falls between the snapshot and this one."""
    chance = (rows["snap_chance_of_playing"] / 100).fillna(1.0)
    about_this_gw = rows["snap_time"].notna() & (
        rows["prev_deadline"].isna() | (rows["snap_time"] > rows["prev_deadline"]))
    avail = chance.where(about_this_gw, 1.0)
    return avail.where(rows["snap_status"] != "u", 0.0)


def predict_fixtures(pf: pd.DataFrame, gameweeks: pd.DataFrame, scoring: dict,
                     prior_90s: dict | float = PRIOR_90S) -> pd.DataFrame:
    """Drivers and components for EVERY player_fixture row (played and future).

    Played rows get the prediction that would have been made at that
    gameweek's deadline (the walk-forward evaluation uses them); future rows
    get today's. Returns one row per player per fixture.

    prior_90s: shrinkage strength per stat and position (PRIOR_90S), or one
    number for every stat (used for the sensitivity check)."""
    order = gw_order(gameweeks)
    teams = team_strength(pf, gameweeks)
    own = teams.rename(columns={"att": "att_own", "dfn": "dfn_own"})
    opp = teams[GW + ["team_code", "att", "dfn"]].rename(
        columns={"team_code": "opponent_code", "att": "att_opp", "dfn": "dfn_opp"})
    df = (pf.merge(order[GW + ["prev_deadline"]], on=GW, how="left")
            .merge(position_priors(pf, gameweeks, scoring), on=GW + ["position"], how="left")
            .merge(own, on=GW + ["team_code"], how="left")
            .merge(opp, on=GW + ["opponent_code"], how="left"))
    if "snap_time" not in df:     # built without snapshots (tests, or no data yet)
        df["snap_time"] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
        df["snap_status"], df["snap_chance_of_playing"] = None, np.nan
    pos = df["position"]

    # Minutes: recent rates x availability. No history -> 0, like the v0 baseline.
    w = MIN_WINDOW
    df["availability"] = availability(df)
    df["p_play"] = (df["availability"] * df[f"f_appear_rate_{w}"]).fillna(0.0)
    df["p60"] = (df["availability"] * df[f"f_60_rate_{w}"]).fillna(0.0)
    df["exp_mins"] = (df["availability"] * df[f"f_mins_per_match_{w}"]).fillna(0.0)
    df["n_matches"] = df[f"f_n_matches_{w}"].fillna(0).astype(int)

    # Fixture strength. The player's own rates already reflect his own team, so
    # goals / assists / saves are scaled by the OPPONENT only; the clean sheet
    # needs both sides: lambda_against = opponent attack x own defence / league.
    home = np.where(df["home"], df["home_adv"], 1 / df["home_adv"])
    vs_defence = df["dfn_opp"] / df["league_xg"] * home       # > 1 = leaky opponent
    vs_attack = df["att_opp"] / df["league_xg"] / home        # > 1 = dangerous opponent
    lam_against = df["att_opp"] * df["dfn_own"] / df["league_xg"] / home

    r = RATE_WINDOW
    def m(stat):
        return pos.map(prior_90s[stat]) if isinstance(prior_90s, dict) else prior_90s

    minutes, defcon_minutes = df[f"f_minutes_sum_{r}"], df[f"f_defcon_minutes_{r}"]
    df["xg90"] = shrink(df[f"f_xg_per90_{r}"], minutes, df["prior_xg90"], m("xg"))   # his own rate
    df["xa90"] = shrink(df[f"f_xa_per90_{r}"], minutes, df["prior_xa90"], m("xa"))
    df["lambda_g"] = df["xg90"] * vs_defence                                         # ... in this fixture
    df["lambda_a"] = df["xa90"] * vs_defence
    saves90 = shrink(df[f"f_saves_per90_{r}"], minutes, df["prior_saves90"], m("saves")) * vs_attack
    defcon90 = shrink(df[f"f_defcon_per90_{r}"], defcon_minutes, df["prior_defcon90"], m("defcon"))
    df["lambda_against"] = lam_against
    df["p_cs"] = np.exp(-lam_against)

    dc = scoring["defensive_contribution"]
    threshold = pos.map(dc["threshold"])                       # NaN for GK: not eligible
    has_defcon = threshold.notna() & defcon90.notna()
    df["p_defcon"] = 0.0
    df.loc[has_defcon, "p_defcon"] = prob_at_least(defcon90[has_defcon].to_numpy(),
                                                   threshold[has_defcon].to_numpy())

    exp_goals = df["lambda_g"] * df["exp_mins"] / 90
    exp_assists = df["lambda_a"] * df["exp_mins"] / 90
    p_sub = df["p_play"] - df["p60"]                           # plays, but under 60
    gc = scoring["goals_conceded"]
    concedes = pos.isin(gc["positions"])
    is_gk = pos.eq("GK")
    defcon_live = df["season"] >= dc["first_season"]          # no DefCon points before 2025-26

    comp = pd.DataFrame(index=df.index)
    comp["xp_appear"] = scoring["appearance"]["under_60"] * p_sub + scoring["appearance"]["over_60"] * df["p60"]
    comp["xp_goals"] = pos.map(scoring["goal"]) * exp_goals
    comp["xp_assists"] = scoring["assist"] * exp_assists
    comp["xp_cs"] = pos.map(scoring["clean_sheet"]) * df["p60"] * df["p_cs"]
    comp["xp_saves"] = np.where(is_gk, scoring["saves"]["points"] * df["p60"]
                                * expected_floor_div(saves90.fillna(0), scoring["saves"]["per"]), 0.0)
    comp["xp_conceded"] = np.where(concedes, gc["points"] * df["p60"]
                                   * expected_floor_div(lam_against.fillna(0), gc["per"]), 0.0)
    comp["xp_defcon"] = np.where(defcon_live, dc["points"] * df["p60"] * df["p_defcon"], 0.0)
    comp["xp_bonus"] = (df["b_60"] * df["p60"] + df["b_sub"] * p_sub + df["b_goal"] * exp_goals
                        + df["b_assist"] * exp_assists + df["b_cs"] * df["p60"] * df["p_cs"])
    comp["xp_cards"] = df["card_pts90"] * df["exp_mins"] / 90
    # NaN only where nothing is known yet (no earlier data at all): predict 0.
    df[COMPONENTS] = comp.fillna(0.0)
    df["opponent"] = df["opponent_code"]
    return df


def to_gameweeks(fx: pd.DataFrame) -> pd.DataFrame:
    """Per-fixture predictions -> one row per (season, player_code, gw).

    Components and exp_mins are SUMMED (a double counts both matches).
    Probabilities and per-90 rates are AVERAGED over the fixtures, so they
    still read as "per match"; opponent and home become lists, one entry per
    fixture. xp is the sum of the gameweek's components, so it equals them
    exactly (no rounding drift)."""
    keys = ["season", "player_code", "gw"]
    by = fx.sort_values("kickoff_time").groupby(keys)
    out = by[COMPONENTS + ["exp_mins"]].sum()
    out[["p_play", "p60", "lambda_g", "lambda_a", "p_cs", "p_defcon"]] = by[
        ["p_play", "p60", "lambda_g", "lambda_a", "p_cs", "p_defcon"]].mean()
    out["opponent"] = by["opponent"].agg(list)
    out["home"] = by["home"].agg(list)
    out["n_fixtures"] = by.size()
    out[["team_code", "position", "n_matches", "availability"]] = by[
        ["team_code", "position", "n_matches", "availability"]].first()
    out["xp"] = out[COMPONENTS].sum(axis=1)
    return out.reset_index()


def upcoming(gw_pred: pd.DataFrame, roster: pd.DataFrame, season: str, first_gw: int,
             last_gw: int, horizon: int = HORIZON) -> pd.DataFrame:
    """One row per roster player per gameweek first_gw .. first_gw + horizon - 1
    (cut at the season's end), blanks included: n_fixtures 0, every component
    0, no opponent."""
    gws = list(range(first_gw, min(first_gw + horizon - 1, last_gw) + 1))
    grid = (roster[["player_code", "team_code", "position"]]
            .merge(pd.DataFrame({"gw": gws}), how="cross").assign(season=season))
    cols = [c for c in gw_pred.columns if c not in ("team_code", "position")]
    out = grid.merge(gw_pred[cols], on=["season", "player_code", "gw"], how="left")
    blank = out["n_fixtures"].isna()
    out.loc[blank, COMPONENTS + ["xp", "exp_mins"]] = 0.0
    out["n_fixtures"] = out["n_fixtures"].fillna(0).astype(int)
    out["n_matches"] = out["n_matches"].fillna(0).astype(int)
    for col in ["opponent", "home"]:
        out[col] = [v if isinstance(v, list) else [] for v in out[col]]
    first = ["season", "player_code", "gw", "team_code", "position", "n_fixtures"]
    rest = ["n_matches", "availability", "xp", *COMPONENTS,
            *[d for d in DRIVERS if d not in ("opponent", "home")], "opponent", "home"]
    return out[first + rest].sort_values(["gw", "player_code"]).reset_index(drop=True)


def main() -> None:
    pf = pd.read_parquet(PROCESSED / "player_fixture.parquet")
    gameweeks = pd.read_parquet(PROCESSED / "gameweeks.parquet")
    players = pd.read_parquet(PROCESSED / "players.parquet")
    players_daily = pd.read_parquet(PROCESSED / "players_daily.parquet")

    # "Now" = the latest snapshot, as in predict_baseline: same raw data, same table.
    as_of = players_daily["snapshot_time"].max()
    season_gws = gameweeks[gameweeks["season"] == CURRENT_SEASON]
    later = season_gws[season_gws["deadline_time"] > as_of]
    if later.empty:
        print("xp_components: no gameweek deadline left this season, nothing to predict")
        return
    first_gw, last_gw = int(later["gw"].min()), int(season_gws["gw"].max())

    fx = predict_fixtures(pf, gameweeks, load_scoring())
    fx = fx[fx["is_future"] & (fx["season"] == CURRENT_SEASON)]
    roster = players[players["season"] == CURRENT_SEASON]
    out = upcoming(to_gameweeks(fx), roster, CURRENT_SEASON, first_gw, last_gw)
    out["as_of"] = as_of
    out.to_parquet(PROCESSED / "xp_components.parquet", index=False)
    print(f"xp_components {out.shape}  GW{out['gw'].min()}-{out['gw'].max()}  "
          f"as of {as_of:%Y-%m-%d %H:%M} UTC")


if __name__ == "__main__":
    main()
