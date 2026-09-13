"""Print mean +/- sample standard deviation for validation-only Phase 1 runs."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT.parent / "outputs"
METRIC_KEYS = ("mae_k", "rmse_k", "r2", "bias_k", "p95_absolute_error_k")


def mean_std(values: list[float]) -> str:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.6f} +/- {std:.6f}"


def main() -> None:
    rows: list[tuple[str, int, list[str]]] = []
    for experiment_dir in sorted(path for path in OUTPUT_DIR.iterdir() if path.is_dir()):
        metric_files = sorted((experiment_dir / "validation_only").glob("seed_*/metrics.json"))
        if not metric_files:
            continue
        records = [json.loads(path.read_text(encoding="utf-8")) for path in metric_files]
        if any(record.get("test_evaluated") for record in records):
            raise ValueError(f"Test-evaluated result found in validation-only folder: {experiment_dir}")
        summaries = [
            mean_std([float(record["validation_metrics"][key]) for record in records])
            for key in METRIC_KEYS
        ]
        rows.append((experiment_dir.name, len(records), summaries))

    if not rows:
        raise FileNotFoundError(f"No validation-only metrics found below {OUTPUT_DIR}")

    header = ["experiment", "n", *METRIC_KEYS]
    widths = [len(item) for item in header]
    printable = [[name, str(count), *summaries] for name, count, summaries in rows]
    for row in printable:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    print("  ".join(value.ljust(width) for value, width in zip(header, widths)))
    print("  ".join("-" * width for width in widths))
    for row in printable:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))


if __name__ == "__main__":
    main()
