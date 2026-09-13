# Figure QA notes

## Figure contract

- Core conclusion: Complete-test RMSE decreases as lookback increases from 15 to about 40 samples, then enters a shallow plateau; the numerical minimum is lookback 60.
- Results-level question: How much historical context is useful for the 15-step conditional Thv forecast on the complete 0715-BACK sequence?
- Archetype: Single-panel quantitative trend.
- Backend: Python/Matplotlib only.
- Final size: 7.20 × 3.45 in (approximately 183 mm wide).
- Center and spread: Mean ± one sample SD across 10 random seeds.
- Source data: `lookback_test_summary.csv` and `lookback_test_all_runs.csv` produced by frozen-checkpoint inference.

## Data integrity

- Models: 12 lookbacks × 10 seeds = 120 frozen GRU checkpoints.
- Forecast horizon: 15 samples (150 s).
- Test split: Original 0715-BACK complete sequence.
- Common forecast origins: 10,659 per run, all requiring 120 historical samples.
- Exclusions for plotting: none; every trained lookback and every seed is shown.
- Training or parameter updates during test evaluation: none.
- Numerical minimum: lookback 60, mean full-test RMSE 1.180278 K.
- Plateau diagnostic retained in the analysis record but not printed on the figure: lookback 40–120; mean range 0.013202 K.

## Methodological caveat

Because all lookbacks are compared on 0715-BACK, this run is now a post-hoc lookback evaluation/selection set. It cannot simultaneously be described as an untouched final holdout for a model selected using this figure.

## Rendered QA

- Source preflight: PASS (21 checks, 0 warnings, 0 failures).
- Panel alignment: NOT APPLICABLE/PASS for a single panel; alignment JSON retained.
- PDF glyph floor: PASS; minimum rendered text size 6.8 pt.
- Collision audit: PASS; 0 failures and 0 warnings.
- Visual inspection: all 120 seed points are visible; the lowest point is not clipped; mean line, SD bars, minimum annotation and legend are legible at final size.

## Provenance

The previous rapid-validation version was copied to `archive_validation_rapid_20260905` before the requested filename was overwritten.
