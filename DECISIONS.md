# DECISIONS.md

What was tried, why, and what happened, with real numbers. Newest first.

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
