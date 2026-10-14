# DECISIONS.md

What was tried, why, and what happened, with real numbers. Newest first.

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
