"""Where does v1's ranking skill live, and is the FPL bar fair? (DECISIONS 2026-11-24)

Run from the project root: python src/diagnose_spearman.py

1. Spread: within each (season, gw, position), the share of the variance of
   actual points that comes from each component: cov(component, total) / var(total).
   The shares sum to 1 apart from penalties and own goals.
2. Minutes oracle: v1 re-run with its minutes drivers replaced by what really
   happened in that match. NOT a model (it leaks on purpose): it is the ceiling
   a perfect minutes model could reach with everything else unchanged.
3. Point-in-time FPL: on 2026-27 (GW6 on), v1 vs FPL's ep_next from the snapshot
   strictly before each deadline. vaastav's fpl_xp has an undocumented capture
   time, so this is the only fair comparison with FPL we have.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from build_player_match import CURRENT_SEASON
from evaluate_components import KEYS, actual_components, last5_baseline, within_position_spearman
from predict_components import COMPONENTS, MIN_WINDOW, load_scoring, predict_fixtures, to_gameweeks

PROCESSED = Path("data/processed")


def spread_shares(act: pd.DataFrame) -> pd.Series:
    """Share of var(total_points) from each component; `act` already demeaned."""
    v = act["total_points"].var()
    return pd.Series({c.replace("xp_", ""): act[c].cov(act["total_points"]) / v for c in COMPONENTS})


def with_minutes_oracle(pf: pd.DataFrame, exact: bool) -> pd.DataFrame:
    """Overwrite the minutes features of played fixtures with the real outcome.
    exact=False: knows only WHETHER he plays (what good team news tells you);
    60+ and minutes stay his usual rate given that he plays."""
    w, out = MIN_WINDOW, pf.copy()
    played, m = ~pf["is_future"], pf["minutes"]
    plays = (m > 0).astype(float)
    if exact:
        p60, mins = (m >= 60).astype(float), m
    else:
        rate = pf[f"f_appear_rate_{w}"]
        p60 = plays * (pf[f"f_60_rate_{w}"] / rate).fillna(0.7)
        mins = plays * (pf[f"f_mins_per_match_{w}"] / rate).fillna(60)
    out.loc[played, f"f_appear_rate_{w}"] = plays
    out.loc[played, f"f_60_rate_{w}"] = p60
    out.loc[played, f"f_mins_per_match_{w}"] = mins
    return out


def main() -> None:
    pf = pd.read_parquet(PROCESSED / "player_fixture.parquet")
    pg = pd.read_parquet(PROCESSED / "player_gw.parquet")
    gameweeks = pd.read_parquet(PROCESSED / "gameweeks.parquet")
    scoring = load_scoring()
    pg["last5"] = last5_baseline(pg)

    def v1(frame):
        return to_gameweeks(predict_fixtures(frame, gameweeks, scoring).loc[lambda d: ~d["is_future"]])[KEYS + ["xp"]]

    def report(r, col, label):
        pl = r[r["minutes"] > 0]
        print(f"  {label:34s} Spearman all {within_position_spearman(r, col):.3f}   "
              f"played {within_position_spearman(pl, col):.3f}   MAE {(r[col] - r['total_points']).abs().mean():.3f}")

    # The bar rows of evaluate_components (past seasons, fpl_xp recorded).
    rows = pg[(pg["n_fixtures"] > 0) & pg["fpl_xp"].notna() & pg["last5"].notna()]
    rows = rows[KEYS + ["position", "minutes", "total_points", "fpl_xp"]]
    print(f"Bar rows: {len(rows):,}\n")

    # ---- 1. spread
    act = rows.merge(actual_components(pf[~pf["is_future"]], scoring), on=KEYS)
    cols = COMPONENTS + ["total_points"]
    dem = act[cols] - act.groupby(["season", "gw", "position"])[cols].transform("mean")
    tab = pd.DataFrame({"all": spread_shares(dem), "played": spread_shares(dem[act["minutes"] > 0])})
    for pos in ["GK", "DEF", "MID", "FWD"]:
        tab[pos] = spread_shares(dem[act["position"] == pos])
    print("1. Share of within-position variance of actual points (%)")
    print((tab * 100).round(1).to_string(), "\n")

    # ---- 2. minutes oracle
    print("2. Minutes oracle (leaks on purpose: a ceiling, not a model)")
    for label, frame in [("v1 as built", pf), ("+ knows whether he plays", with_minutes_oracle(pf, exact=False)),
                         ("+ knows his exact minutes", with_minutes_oracle(pf, exact=True))]:
        report(rows.merge(v1(frame), on=KEYS), "xp", label)
    report(rows, "fpl_xp", "FPL fpl_xp (capture time unknown)")
    print()

    # ---- 3. point-in-time FPL
    r = pg[(pg["season"] == CURRENT_SEASON) & (pg["n_fixtures"] > 0)
           & pg["snap_ep_next"].notna() & pg["last5"].notna()]
    r = r[KEYS + ["position", "minutes", "total_points", "snap_ep_next", "last5"]].merge(
        v1(pf).rename(columns={"xp": "v1"}), on=KEYS)
    print(f"3. {CURRENT_SEASON}, point-in-time ep_next: {len(r):,} rows, GW {r['gw'].min()}-{r['gw'].max()}")
    for col in ["last5", "v1", "snap_ep_next"]:
        report(r, col, col)
    per_gw = pd.Series({gw: within_position_spearman(d, "v1") - within_position_spearman(d, "snap_ep_next")
                        for gw, d in r.groupby("gw")})
    rng = np.random.default_rng(0)
    boot = [rng.choice(per_gw.to_numpy(), len(per_gw)).mean() for _ in range(10_000)]
    print(f"  v1 - ep_next per GW: {per_gw.round(3).to_dict()}")
    print(f"  mean {per_gw.mean():+.3f}, {(per_gw > 0).sum()}/{len(per_gw)} GWs positive, "
          f"bootstrap over GWs 95% CI [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]")


if __name__ == "__main__":
    main()
