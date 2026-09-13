"""Summarize validation-only Phase 2 runs as mean +/- sample standard deviation."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT.parent / "outputs_phase2"
OVERALL_METRIC_KEYS = ("mae_k", "rmse_k", "p95_absolute_error_k")
EVENT_METRIC_KEYS = ("mae_k", "rmse_k", "p95_absolute_error_k")


def mean_std(values: list[float]) -> str:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.6f} +/- {std:.6f}"


def main() -> None:
    if not OUTPUT_DIR.exists():
        raise FileNotFoundError(f"Phase 2 output directory does not exist: {OUTPUT_DIR}")
    rows: list[list[str]] = []
    for experiment_dir in sorted(path for path in OUTPUT_DIR.iterdir() if path.is_dir()):
        files = sorted((experiment_dir / "validation_only").glob("seed_*/metrics.json"))
        if not files:
            continue
        records = [json.loads(path.read_text(encoding="utf-8")) for path in files]
        if any(record.get("test_evaluated") for record in records):
            raise ValueError(f"Test result found in validation-only folder: {experiment_dir}")
        event_metrics = [
            mean_std(
                [float(record["validation_history_event_metrics"][key]) for record in records]
            )
            for key in EVENT_METRIC_KEYS
        ]
        overall_metrics = [
            mean_std([float(record["validation_metrics"][key]) for record in records])
            for key in OVERALL_METRIC_KEYS
        ]
        parameters = {int(record["parameter_count"]) for record in records}
        if len(parameters) != 1:
            raise ValueError(f"Parameter count changed across seeds: {experiment_dir}")
        rows.append(
            [
                experiment_dir.name,
                str(len(records)),
                f"{parameters.pop():,}",
                *event_metrics,
                *overall_metrics,
            ]
        )

    if not rows:
        raise FileNotFoundError(f"No Phase 2 validation metrics found below {OUTPUT_DIR}")
    header = [
        "experiment",
        "n",
        "parameters",
        "event_mae_k",
        "event_rmse_k",
        "event_p95_k",
        "overall_mae_k",
        "overall_rmse_k",
        "overall_p95_k",
    ]
    widths = [len(item) for item in header]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    print("  ".join(value.ljust(width) for value, width in zip(header, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))


if __name__ == "__main__":
    main()
