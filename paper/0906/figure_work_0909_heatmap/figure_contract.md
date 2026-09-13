# Figure contract

- Core conclusion: Parallel graph-fusion models reduce full-test RMSE consistently across random seeds, whereas serial graph fusion is less accurate and less stable.
- Results-level question: Is the architecture ranking reproducible across initialization seeds rather than driven by a small number of favorable runs?
- Archetype: Single-panel quantitative heatmap.
- Backend: Python/matplotlib.
- Final size: 183 mm × approximately 138 mm.
- Evidence: Every available seed-level full-test RMSE for the retained baseline, parallel and serial architectures; the exploratory TCN is excluded at the user's request.
- Statistics: Cell-level RMSE for ten matched random seeds per retained architecture.
- Missingness: None among retained architectures.
- Reviewer risk: The heatmap is a robustness visualization, not a replacement for paired tests or mean-and-uncertainty comparisons.
