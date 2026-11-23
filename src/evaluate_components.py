"""Walk-forward evaluation of the v1 decomposed xP against the Phase 1 bars.

Run from the project root (after build_all.py): python src/evaluate_components.py

Rows: the same 32,068 player-gameweeks as the 2026-10-14 bar (DECISIONS.md):
played (n_fixtures > 0), FPL's own prediction recorded (fpl_xp not NaN), and
at least one earlier played gameweek (so the last-5 baseline exists).

Walk-forward for free: predict_fixtures gives every PLAYED fixture the
prediction that would have been made at that gameweek's deadline (features
and every estimated constant frozen there), 1 gameweek ahead. Nothing is fit
on the gameweek being predicted.

Metrics: MAE, within-position Spearman (rank correlation inside each
(season, gw, position), averaged; the bar's definition), Pearson. Reported on
all rows and on rows where the player actually played (minutes > 0), since
"predicted ~0, scored 0" flatters every metric.

Caveat (DATA.md): the past schedule is the end-of-season one, so a backtest
knows about doubles before they were announced. Fine for comparing models on
the same rows, not for a headline backtest.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from predict_components import COMPONENTS, load_scoring, predict_fixtures, to_gameweeks

PROCESSED = Path("data/processed")
KEYS = ["season", "player_code", "gw"]
GROUP = ["season", "gw", "position"]


def last5_baseline(player_gw: pd.DataFrame) -> pd.Series:
    """The bar's baseline: mean points over the previous 5 played gameweeks
    (shifted, so it never sees the gameweek it predicts) x n_fixtures."""
    df = player_gw.sort_values(["player_code", "season", "gw"])
    played = df[df["n_fixtures"] > 0]
    mean5 = played.groupby("player_code")["total_points"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    return (mean5 * played["n_fixtures"]).reindex(player_gw.index)


def within_position_spearman(df: pd.DataFrame, col: str, target: str = "total_points") -> float:
    """Rank both inside each (season, gw, position), Pearson of the ranks per
    group, averaged over groups (groups where either side is constant are skipped)."""
    ranks = df[GROUP].assign(a=df.groupby(GROUP)[col].rank(), b=df.groupby(GROUP)[target].rank())
    per_group = ranks.groupby(GROUP)[["a", "b"]].apply(lambda g: g["a"].corr(g["b"]))
    return float(per_group.mean())


def metrics(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: {"MAE": (df[c] - df["total_points"]).abs().mean(),
                             "Spearman (within pos)": within_position_spearman(df, c),
                             "Pearson": df[c].corr(df["total_points"])} for c in cols}).T


def actual_components(pf: pd.DataFrame, scoring: dict) -> pd.DataFrame:
    """What each component actually scored, per player-gameweek, from the
    match outcomes and scoring.yaml: the target each xp_* tries to predict."""
    s, pos = scoring, pf["position"]
    m = pf["minutes"]
    dc = s["defensive_contribution"]
    out = pf[KEYS].copy()
    out["xp_appear"] = np.select([m >= 60, m > 0], [s["appearance"]["over_60"], s["appearance"]["under_60"]], 0)
    out["xp_goals"] = pos.map(s["goal"]) * pf["goals_scored"]
    out["xp_assists"] = s["assist"] * pf["assists"]
    out["xp_cs"] = pos.map(s["clean_sheet"]) * pf["clean_sheets"] * (m >= 60)
    out["xp_saves"] = np.where(pos.eq("GK"), s["saves"]["points"] * (pf["saves"] // s["saves"]["per"]), 0)
    concedes = pos.isin(s["goals_conceded"]["positions"])
    out["xp_conceded"] = np.where(concedes, s["goals_conceded"]["points"]
                                  * (pf["goals_conceded"] // s["goals_conceded"]["per"]), 0)
    hit = pf["defensive_contribution"] >= pos.map(dc["threshold"])
    out["xp_defcon"] = np.where(hit & (pf["season"] >= dc["first_season"]), dc["points"], 0)
    out["xp_bonus"] = pf["bonus"]
    out["xp_cards"] = s["cards"]["yellow"] * pf["yellow_cards"] + s["cards"]["red"] * pf["red_cards"]
    return out.groupby(KEYS)[COMPONENTS].sum().reset_index()


def main() -> None:
    pf = pd.read_parquet(PROCESSED / "player_fixture.parquet")
    player_gw = pd.read_parquet(PROCESSED / "player_gw.parquet")
    gameweeks = pd.read_parquet(PROCESSED / "gameweeks.parquet")
    scoring = load_scoring()

    played_pf = pf[~pf["is_future"]]
    pred = to_gameweeks(predict_fixtures(pf, gameweeks, scoring).loc[lambda d: ~d["is_future"]])
    player_gw["last5"] = last5_baseline(player_gw)
    rows = player_gw[(player_gw["n_fixtures"] > 0) & player_gw["fpl_xp"].notna()
                     & player_gw["last5"].notna()]
    rows = rows[KEYS + ["position", "minutes", "total_points", "fpl_xp", "last5"]].merge(
        pred[KEYS + ["xp"]].rename(columns={"xp": "v1"}), on=KEYS, how="left")
    assert rows["v1"].notna().all() and len(rows) == 32068, len(rows)

    pd.set_option("display.width", 200)
    cols = ["last5", "fpl_xp", "v1"]
    print(f"All bar rows ({len(rows):,}):")
    print(metrics(rows, cols).round(3), "\n")
    played = rows[rows["minutes"] > 0]
    print(f"Rows with minutes > 0 ({len(played):,}):")
    print(metrics(played, cols).round(3), "\n")

    print("MAE by position:")
    print(rows.groupby("position")[cols].apply(
        lambda g: (g.sub(rows.loc[g.index, "total_points"], axis=0)).abs().mean()).round(3), "\n")

    actual = actual_components(played_pf, scoring)
    both = rows[KEYS].merge(pred[KEYS + COMPONENTS], on=KEYS).merge(
        actual, on=KEYS, suffixes=("", "_actual"))
    calib = pd.DataFrame({"predicted": both[COMPONENTS].mean(),
                          "actual": both[[f"{c}_actual" for c in COMPONENTS]].mean().to_numpy()})
    calib.loc["total"] = calib.sum()
    calib.loc["total_points (incl. pens, own goals)"] = [np.nan, rows["total_points"].mean()]
    print("Mean points per player-gameweek, by component (calibration):")
    print(calib.round(3), "\n")

    # Sensitivity, not tuning: m was set from split-half reliability before
    # looking at these numbers; this only shows how much the choice matters.
    print("Shrinkage strength m (all bar rows):")
    variants = {"no shrinkage (m ~ 0)": 1e-6, "one m = 6 for every stat": 6.0,
                "per-stat m (PRIOR_90S, used)": None}
    for name, m in variants.items():
        fx = predict_fixtures(pf, gameweeks, scoring) if m is None else \
            predict_fixtures(pf, gameweeks, scoring, prior_90s=m)
        gw = to_gameweeks(fx.loc[~fx["is_future"]])
        r = rows[KEYS + ["position", "total_points"]].merge(gw[KEYS + ["xp"]], on=KEYS)
        print(f"  {name:30s} MAE {(r['xp'] - r['total_points']).abs().mean():.3f}  "
              f"Spearman {within_position_spearman(r, 'xp'):.3f}")


if __name__ == "__main__":
    main()
