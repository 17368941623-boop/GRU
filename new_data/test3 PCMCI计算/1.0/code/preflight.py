#!/usr/bin/env python3
"""Check the server layout and the exact data scope before PCMCI starts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_io import load_original_training_records
from protocol import DEFAULT_MAX_LAG, expand_static_candidates, load_raw_columns


def main() -> None:
    code_dir = Path(__file__).resolve().parent
    experiment_dir = code_dir.parent
    default_data = experiment_dir.parent / "processed_data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--output", type=Path, default=experiment_dir / "output" / "preflight.json")
    parser.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    raw_columns = load_raw_columns(data_dir)
    records, data_audit = load_original_training_records(
        data_dir / "train_clean.pkl", raw_columns, min_length=args.max_lag + 40
    )
    physical, metadata = expand_static_candidates(raw_columns, args.max_lag)
    parent_rows: dict[str, int] = {}
    for record in records:
        parent_rows[str(record["parent_id"])] = parent_rows.get(str(record["parent_id"]), 0) + len(record["values"])
    payload = {
        "status": "PASS",
        "data_dir": str(data_dir),
        "train_file_read": str(data_dir / "train_clean.pkl"),
        "validation_files_read": False,
        "test_files_read": False,
        "raw_variables": len(raw_columns),
        "raw_columns": list(raw_columns),
        "max_lag_steps": args.max_lag,
        "all_lagged_correlations_expected": len(raw_columns) ** 2 * args.max_lag,
        "physical_candidates_including_H_mix_proxies": len(physical),
        "physical_candidate_types": sorted({value["physical_relation_type"] for value in metadata.values()}),
        "parent_rows": parent_rows,
        **data_audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

