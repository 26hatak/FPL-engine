# DECISIONS.md

What was tried, why, and what happened, with real numbers. Newest first.

## 2026-11-23: dropped vaastav's duplicate rows

**What:** vaastav's 2025-26 `merged_gw` repeats 10 rows exactly (2 players), so `player_gw` counted those matches twice. `build_player_match` now drops exact copies, and a real-data test checks one row per player per match.

**Effect** (`python src/evaluate_components.py`, same bar rows):

| Model | MAE before → after | Within-position Spearman |
| --- | --- | --- |
| last-5 | 1.079 → 1.078 | 0.705 (same) |
| `fpl_xp` | 0.855 → 0.854 | 0.759 (same) |
| v1 | 1.015 → 1.014 | 0.720 (same) |

Too small to change any decision. The bars quoted elsewhere stay valid to 3 decimals except MAE (−0.001).

## 2026-11-22: v1 decomposed xP, no ML (`xp_components`)

**Question:** can xP be a sum of scoring pieces, each driven by a visible number, without losing accuracy? The app needs the breakdown, and later ML should only change how a driver is estimated.

**What:**
- `scoring.yaml`: 2026/27 points per event, checked against the official rules page on 2026-11-22. GK goal = 10, CS: GK/DEF 4, MID 1, FWD 0. DefCon: DEF 10+ CBIT, MID/FWD 12+ CBIRT, 2 points, GK not eligible.
- `src/build_player_fixture.py` → `player_fixture`: one row per player per fixture, played and future (82,464 rows, 18,872 future). `f_*`/`snap_*` come from `build_player_gw`'s own functions: future gameweeks are added as empty rows, so they carry the state at the last deadline. Both fixtures of a double share them. Opponent, home and rest days are per fixture.
- `src/predict_components.py` → `xp_components` (same players × GW12–17 as `xp`). xp = appear + goals + assists + CS + saves + conceded + DefCon + bonus + cards.

| Driver | v1 estimate | Why |
| --- | --- | --- |
| `p_play`, `p60`, `exp_mins` | Share of the last 5 matches with minutes / with 60+, minutes per match; × availability (next GW only, `u` = 0 throughout) | Same signal and window as the v0 baseline, but split, because 1 vs 2 appearance points and the CS gate need P(60+), not mean minutes |
| `xg90`, `xa90` | Last 10 GWs per 90, shrunk to the position mean: (events + m·prior) / (90s + m) | Empirical Bayes: one big match says little. m is set from split-half reliability (below), not tuned |
| `lambda_g`, `lambda_a` | `xg90` × opponent's xG conceded / league average × home factor | The player's own rate already contains his own team, so only the opponent adjusts it |
| `p_cs` | exp(−λ_against), λ_against = opponent attack × own defence / league average ÷ home factor; team xG over the last 10 matches, shrunk by 5 matches | Poisson goals, Dixon–Coles-style multiplicative ratings |
| saves, conceded | E[floor(Poisson/3)] and E[floor(Poisson/2)] | "1 point per 3 saves" is not λ/3: 2 saves score nothing |
| `p_defcon` | P(Poisson(shrunk DefCon/90) ≥ threshold \| 60+) | The threshold is a step, so a rate × points would be wrong |
| bonus | Least squares of bonus on [60+, 1–59 min, goals, assists, CS] per position, no intercept | "Average bonus given the events", linear, so it applies to expectations |
| cards | Position average card points per 90 × `exp_mins` | Small and noisy; a player-level rate isn't worth it yet |

**Leakage:** every estimated constant (team ratings, league average, home factor, position priors, bonus coefficients, shrinkage priors) is summed per GW, accumulated, then shifted one GW, the same rule as `f_*`. `tests/test_components.py::test_no_leakage_perturbation` poisons one GW's outcomes and checks that nothing at or before it moves. I checked that it can fail: removing either shift turns it red.

**Shrinkage strength m** (2024-25, players with 450+ minutes in both halves; m = 90s per half × (1 − ρ) / ρ):

| Stat | DEF | MID | FWD | GK |
| --- | --- | --- | --- | --- |
| xG | 28 (ρ 0.30) | 4 (0.73) | 13 (0.47) | – |
| xA | 3 (0.78) | 5 (0.70) | 8 (0.60) | – |
| Saves | – | – | – | 18 (0.42) |
| DefCon (2025-26) | 3 (0.81) | 3 (0.80) | 6 (0.62) | – |

A defender's xG is mostly noise (set pieces), while DefCon is a stable trait. The prior's strength should differ by stat, and the data says by how much. Plain method of moments failed (negative variances): xG is a sum of small shot values, far less noisy than a Poisson count.

**Results.** Walk-forward, 1 GW ahead, on the same 32,068 rows as the 2026-10-14 bar (`python src/evaluate_components.py`):

| Rows | Model | MAE | Within-position Spearman | Pearson |
| --- | --- | --- | --- | --- |
| All 32,068 | last-5 | 1.079 | 0.705 | 0.493 |
| | **v1** | **1.015** | **0.720** | **0.545** |
| | `fpl_xp` | 0.855 | 0.759 | 0.731 |
| Minutes > 0 (13,523) | last-5 | 2.109 | 0.265 | 0.254 |
| | **v1** | **1.931** | **0.306** | **0.301** |
| | `fpl_xp` | 1.653 | 0.558 | 0.607 |

Calibration (mean points per player-GW, predicted vs actual): appear 0.720/0.712, goals 0.195/0.186, assists 0.079/0.105, CS 0.171/0.171, saves 0.019/0.018, conceded −0.066/−0.066, DefCon 0.021/0.023, bonus 0.087/0.088, cards −0.068/−0.061; total 1.158/1.175.

Shrinkage sensitivity: no shrinkage 1.014 / 0.718, one m = 6 1.015 / 0.720, per-stat m 1.015 / 0.720.

**Decision:** keep `xp_components` as the Phase 1 base and the app's breakdown. `xp` (v0) stays the optimizer's input until the MILP session chooses to switch.

**What this tells us:**
- Decomposing beats last-5 on every metric, with no ML, and the pieces add up to the right total. The structure works.
- **The gap to FPL is minutes.** Among players who played, FPL ranks at 0.558 and we rank at 0.306. Each event component is calibrated, so the error is who plays. That's the minutes model, next on the plan as the most important piece. Past seasons have no availability flags, so part of FPL's edge is team news we can't replay.
- Assists are 25% under-predicted: FPL awards assists more generously than Opta xA counts them (rebounds, deflections, penalties won). Fix: scale xA to FPL assists per position.
- Shrinkage barely moves averaged metrics, because most rows are decided by minutes. Its job is the top of the table: unshrunk, the best xG/90 this week is Mundle (1 minute played, 15.3); shrunk, it's Barry/Haaland (0.57/0.54), and Hinshelwood (157 min, 1.44) falls to 0.55. The optimizer reads the top of the table, so shrinkage is the guard against the optimizer's curse.

**Caveats:**
- Same as the bar: the past schedule is end-of-season, so doubles are known early. The m values were estimated on 2024-25, which is also in the evaluation rows: 10 numbers, a small leak, flagged.
- Bonus coefficients are dominated by the old BPS. 2026/27 reworked it, so refit on this season only once it has enough matches.
- Penalties and own goals are in `scoring.yaml` but not modelled (≈0.01 points per match).
- No standard deviation yet.
- vaastav 2025-26 repeats 10 rows exactly (2 players), so `player_gw` double-counts those matches. `player_fixture` drops the duplicates, but `build_player_match` should too (follow-up).

## 2026-11-05: v0 xP = last-5 points per match (stage 1 of the end-to-end gate)

**Question:** what is the simplest xP table the wildcard optimizer can run on, so all stages run by the Nov 8 gate?

**What:** `src/predict_baseline.py` → `data/processed/xp.parquet`, one row per current player per gameweek for the next 6 (first run: 670 players × GW10–15, as of the 2026-11-05 snapshot).

xp = pts_per_match × n_fixtures × availability

| Piece | Definition | Why this and not something else |
| --- | --- | --- |
| `pts_per_match` | Σ points / Σ matches over the player's last 5 gameweeks **with a match**, across seasons | Per match, so a double doesn't inflate the rate. 0-minute matches count, so benched or injured players drop: the only minutes signal v0 has. Blanks are skipped, not counted as zeros. |
| `n_fixtures` | His team's matches in that gameweek, from `fixtures` | A double = 2× and a blank = 0. That's the one thing a 6-GW plan must get right that a single-GW model doesn't care about. |
| `availability` | `chance_of_playing_next_round` / 100 for the **next** gameweek only; 0 for all 6 if status is `u` (left the club or loaned out) | The flag is only about next week. Applying 0% to all 6 would write off a player out for one match. Ignoring it would let the optimizer buy injured players. Players with status `u` are never coming back, so they're 0 throughout. |
| No history | xp = 0 (`n_matches = 0`) | Can't rate him. Phase 1 shrinks to a position average instead. First run: 0 such players, because everyone in the game has a row once his team plays. |

"Now" is the latest snapshot, not the wall clock, so the same raw data always rebuilds the same table. The first predicted gameweek is the first deadline after that snapshot. Runs daily as the last step of `build_all.py`.

**Check: does the function do what we think?** Walk-forward over 2024-25 and 2025-26: for every gameweek, predict using only earlier gameweeks. Club = the player's last row before that gameweek. No availability, because past seasons have no snapshots.

| Rows | MAE | Within-position Spearman |
| --- | --- | --- |
| Same 32,068 rows as the 2026-10-14 bar, 1 GW ahead | 1.071 (bar: 1.079) | 0.705 (bar: 0.705) |

It reproduces the last-5 bar. The small MAE gain comes from per match vs per gameweek in doubles. FPL's own `fpl_xp` is still far better (0.855).

**How fast does it decay with horizon?** 2025-26 GW7–38, the same target rows for every horizon:

| Horizon (GWs ahead) | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| MAE, all played rows (~24.9k) | 1.040 | 1.086 | 1.121 | 1.142 | 1.161 | 1.179 |
| MAE, rows with minutes > 0 (~9.5k) | 2.229 | 2.266 | 2.288 | 2.295 | 2.308 | 2.323 |
| Within-position Spearman | 0.746 | 0.712 | 0.684 | 0.667 | 0.652 | 0.639 |

**Decision:** use it as the optimizer's input for the gate. Replace it in Phase 1, not before.

**What this tells us:**
- Ranking skill falls about 15% from 1 to 6 GWs ahead. That's a real number for the optimizer's discount δ: later gameweeks really are less certain, so 0.85–0.9 is not arbitrary.
- MAE more than doubles on players who actually played (2.23 vs 1.04). Most rows are "predicted ~0, scored 0", which flatters every metric. Phase 1 should report both.
- The top of the table is hot streaks: N. Jackson 7.8 points per match → 46.8 over 6 GWs. The optimizer will chase these. That's the optimizer's curse in practice, and the reason Phase 2 shrinks xP.

**Caveats:**
- The backtest uses today's `fixtures` (the end-of-season schedule for past seasons), so doubles are "known" before they were announced. Fine for checking the function, not for reporting a backtest. See DATA.md.
- No standard deviation yet. The plan's `xp` table wants mean + std; that comes with Phase 1.
- Availability can't be tested walk-forward: there are no flags before 2026-10-07.

## 2026-10-14: Baseline bars for Phase 1

**Question:** what does a points model have to beat?

**Setup:** every played player-gameweek (`n_fixtures > 0`) where FPL's own prediction was recorded: 32,068 rows (2024-25: 24,011; 2025-26: 8,057, the 11 gameweeks vaastav captured). Last-5 baseline = mean points over the player's previous 5 played gameweeks (shifted, so it never sees the gameweek it predicts) × `n_fixtures`.

| Baseline | MAE | Within-position Spearman | Pearson |
| --- | --- | --- | --- |
| FPL's prediction (`fpl_xp`) | 0.855 | 0.759 | 0.731 |
| Last-5 average × fixtures | 1.079 | 0.705 | 0.493 |

Within-position Spearman = rank correlation inside each (season, gameweek, position) group, averaged.

**Decision:** the model must beat MAE 0.855 and Spearman 0.759 on the same rows and metrics.

**Caveats:**
- The rows include many rarely-used players (0 predicted, 0 scored), which flatters every metric. Also report on players with minutes or meaningful ownership.
- `fpl_xp` capture time is undocumented, so it's a baseline only, never a feature.
- With vaastav's fake zeros left in, `fpl_xp` Pearson was 0.51. Treating unrecorded gameweeks as NaN (not 0) was worth +0.22.
