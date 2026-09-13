#!/usr/bin/env python3
"""Summarize overlap between difference-primary and level-sensitivity graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    experiment_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=experiment_dir / "output")
    args = parser.parse_args()
    root = args.output_root.resolve()
    difference_all = root / "difference" / "stable_pcmci_edges_all.csv"
    level_all = root / "level" / "stable_pcmci_edges_all.csv"
    difference_path = (
        difference_all if difference_all.exists()
        else root / "difference" / "stable_pcmci_edges.csv"
    )
    level_path = (
        level_all if level_all.exists()
        else root / "level" / "stable_pcmci_edges.csv"
    )
    if not difference_path.exists() or not level_path.exists():
        print("Comparison skipped: both difference and level stable edge files are required.")
        return
    difference = pd.read_csv(difference_path)
    level = pd.read_csv(level_path)
    keys = ["source", "destination", "lag_steps"]
    merged = difference.merge(level, on=keys, how="outer", suffixes=("_difference", "_level"), indicator=True)
    merged.to_csv(root / "difference_vs_level_edges.csv", index=False)
    difference_set = {tuple(row) for row in difference[keys].itertuples(index=False, name=None)}
    level_set = {tuple(row) for row in level[keys].itertuples(index=False, name=None)}
    union = difference_set | level_set
    payload = {
        "difference_edges": len(difference_set),
        "level_edges": len(level_set),
        "common_exact_lagged_edges": len(difference_set & level_set),
        "exact_edge_jaccard": len(difference_set & level_set) / len(union) if union else None,
        "difference_source_file": difference_path.name,
        "level_source_file": level_path.name,
        "interpretation": (
            "difference is the primary dynamic-change graph; level is a sensitivity analysis. "
            "Agreement strengthens a delay claim, while disagreement identifies a modeling choice to ablate."
        ),
    }
    (root / "difference_vs_level_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
