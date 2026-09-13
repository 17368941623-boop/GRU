# Architecture comparison: figure contract

- Question: Which tested architecture preserves full-sequence forecasting accuracy across the held-out operating run?
- Evidence: 85 completed frozen-checkpoint evaluations; nine architectures; ten random initializations except five for TCN. No excluded runs and no synthetic values.
- Main comparison: complete test-sequence RMSE, bars = arithmetic mean, error bars = sample SD, aligned individual-seed points without jitter.
- Companion comparison: within-seed RMSE differences against the GRU baseline, with paired t-based 95% confidence intervals. All available matching seeds; TCN has five matched pairs. Intervals quantify initialization variability on one fixed test run, not between-run generalization uncertainty. No significance stars; multiplicity-adjusted tests remain in the audit.
- Protocol: Thv at t+15, lookback 60, common-origin lookback 120, Original 0715-BACK test (10,659 valid forecast origins), Original 260501 validation, validation-only checkpoint selection.
- Scope: full set of valid rolling forecast origins, not an open-loop rollout from one initial state and not a rapid-cooling subset.
- Claim boundary: the parallel GRU/MLP/GNN has the lowest observed mean, but the small gap among parallel methods and against GRU is not a demonstrated universal superiority. Serial models show validation/test ranking reversal. The repeatedly inspected test run is not an untouched final holdout.
- Composition: two separate single-panel quantitative comparisons with complementary roles; no multi-panel alignment groups.
- Backend: Python/matplotlib, continuing the user's established Python workflow.
- Export: 183 mm wide; PDF with embedded TrueType text, SVG with editable text, PNG (600 dpi), TIFF (1000 dpi). All glyphs >= 5 pt. Source embeds the real per-seed table for reproducibility.
- QA: independent JSON/CSV and saved-prediction reconciliation; source preflight; PDF glyph and collision audits; full visual inspection. QA artifacts stay outside the final source-and-images folder.
