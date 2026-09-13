# Final figure review

Both final figures passed the rendered collision audit with zero failures and zero warnings. The source preflight passed 20 checks; its width warning is a literal-parser false positive for the expression 183/25.4. Actual PDF widths are 183.00 mm, with heights 122.00 and 118.00 mm. PDF minimum text sizes are 7.5 and 7.1 pt. Both figures contain one plot area, so between-panel alignment is not applicable.

| Figure | Unique role | Summary and uncertainty | Replicate unit | Data integrity | Visual inspection |
|---|---|---|---|---|---|
| Main | Full-test architecture comparison | Mean run-level RMSE ± sample SD; all seed points without jitter | 10 initializations/model, TCN 5, on one fixed operating run | 85/85 observations; no excluded high-error run; zero-based bar axis; persistence shown | All model names, n values, numeric labels, error bars, points and legends fit; no clipped text; colors distinguish architecture families without carrying essential information alone |
| Paired | Size and uncertainty of within-seed differences vs GRU | Mean paired difference and unadjusted Student-t 95% interval | 10 matched seed pairs/model, TCN 5 | Explicit index matching, no dropped observations within a compared model; all eight comparisons | Labels, intervals, zero reference and explanatory notes separated; no significance stars |

The confidence intervals in the companion figure are not multiplicity-adjusted and must not be used to claim a familywise significant advantage. Holm-adjusted paired Wilcoxon comparisons are retained in result_reconciliation.json. In particular, parallel GRU–MLP–GNN vs GRU has Holm-adjusted p=0.22265625; a unique superior parallel architecture is not established.

The model-ranking result is conditional on the given split and shared training configuration. Temporal baselines and graph models are not parameter-count matched. Checkpoint selection was validation-only, but the repeatedly inspected test run has informed subsequent research choices and should not be described as an untouched final confirmatory holdout.

Figures are full-sequence rolling-origin h=15 predictions with updated measurement history, not no-feedback open-loop rollouts. The data reconciliation checks exported artifacts and does not replace a full raw-data/preprocessing leakage audit.

PDF/SVG are editable vector deliverables. PNG is 600 dpi; LZW TIFF is 1000 dpi. The reproduction source embeds the real 85-run source table and needs no access to model checkpoints. No training, model inference, deletion, or mutation of the exported experiment results was performed.
