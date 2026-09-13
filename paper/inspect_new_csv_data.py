#!/usr/bin/env python3
"""Read-only profiler for the newly exported HTC 8300 CSV segments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_DIR = Path(r"C:\Users\Administrator\Desktop\新建文件夹")
OUTPUT_JSON = Path(__file__).resolve().parent / "new_data_readonly_profile.json"


def read_export(path: Path) -> pd.DataFrame:
    attempts = (
        ("utf-16", "\t"),
        ("utf-8-sig", ","),
        ("gb18030", ","),
        ("gb18030", "\t"),
    )
    errors: list[str] = []
    for encoding, separator in attempts:
        try:
            frame = pd.read_csv(path, encoding=encoding, sep=separator, low_memory=False)
            if len(frame.columns) > 1:
                frame.columns = (
                    frame.columns.astype(str)
                    .str.replace('"', "", regex=False)
                    .str.strip()
                )
                return frame
        except Exception as exc:
            errors.append(f"{encoding}/{separator!r}: {exc}")
    raise ValueError(f"Cannot read {path}: {' | '.join(errors)}")


def longest_true_run(mask: pd.Series) -> int:
    array = mask.to_numpy(dtype=bool, copy=False)
    if array.size == 0 or not array.any():
        return 0
    padded = np.r_[False, array, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return int((changes[1::2] - changes[::2]).max())


def profile_file(path: Path) -> dict[str, object]:
    frame = read_export(path)
    time_columns = [column for column in frame.columns if column.endswith(" Time")]
    value_columns = [column for column in frame.columns if column.endswith(" ValueY")]
    if not time_columns:
        raise ValueError(f"No time columns in {path.name}")

    primary_column = time_columns[0]
    primary_time = pd.to_datetime(frame[primary_column], errors="coerce")
    valid_time = primary_time.dropna()
    positive_diffs = valid_time.diff().dt.total_seconds()
    positive_diffs = positive_diffs[positive_diffs > 0]
    median_dt = float(positive_diffs.median()) if not positive_diffs.empty else None

    per_time = {}
    for column in time_columns:
        parsed = pd.to_datetime(frame[column], errors="coerce")
        valid = parsed.dropna()
        per_time[column] = {
            "invalid": int(parsed.isna().sum()),
            "start": valid.min().isoformat() if not valid.empty else None,
            "end": valid.max().isoformat() if not valid.empty else None,
            "duplicates": int(parsed.duplicated(keep=False).sum()),
            "backward_steps": int((parsed.diff().dt.total_seconds() < 0).sum()),
        }

    missing_columns = []
    for column in value_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        missing_count = int(numeric.isna().sum())
        longest_run = longest_true_run(numeric.isna())
        if missing_count:
            missing_columns.append(
                {
                    "column": column,
                    "missing_count": missing_count,
                    "missing_fraction": missing_count / max(len(frame), 1),
                    "longest_run_rows": longest_run,
                }
            )

    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_names": list(frame.columns),
        "time_columns": len(time_columns),
        "value_columns": len(value_columns),
        "primary_time_column": primary_column,
        "primary_start": valid_time.min().isoformat() if not valid_time.empty else None,
        "primary_end": valid_time.max().isoformat() if not valid_time.empty else None,
        "primary_invalid": int(primary_time.isna().sum()),
        "primary_duplicates": int(primary_time.duplicated(keep=False).sum()),
        "primary_backward_steps": int((primary_time.diff().dt.total_seconds() < 0).sum()),
        "median_positive_dt_seconds": median_dt,
        "all_time_columns_rowwise_equal": all(
            frame[column].fillna("<NA>").astype(str).equals(
                frame[primary_column].fillna("<NA>").astype(str)
            )
            for column in time_columns[1:]
        ),
        "per_time_column": per_time,
        "missing_value_columns": missing_columns,
        "completely_missing_value_columns": [
            entry["column"]
            for entry in missing_columns
            if entry["missing_count"] == len(frame)
        ],
    }


def main() -> None:
    files = sorted(SOURCE_DIR.glob("*.csv"))
    profiles = []
    signal_stats: dict[str, dict[str, object]] = {}
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path.name}", flush=True)
        frame = read_export(path)
        profiles.append(profile_file(path))
        for column in [name for name in frame.columns if name.endswith(" ValueY")]:
            numeric = pd.to_numeric(frame[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            valid = numeric.dropna()
            entry = signal_stats.setdefault(
                column,
                {
                    "column": column,
                    "rows": 0,
                    "missing": 0,
                    "zeros": 0,
                    "negative": 0,
                    "minimum": None,
                    "maximum": None,
                    "longest_constant_run": 0,
                },
            )
            entry["rows"] = int(entry["rows"]) + len(numeric)
            entry["missing"] = int(entry["missing"]) + int(numeric.isna().sum())
            entry["zeros"] = int(entry["zeros"]) + int(numeric.eq(0).sum())
            entry["negative"] = int(entry["negative"]) + int(numeric.lt(0).sum())
            if not valid.empty:
                current_min = float(valid.min())
                current_max = float(valid.max())
                entry["minimum"] = (
                    current_min
                    if entry["minimum"] is None
                    else min(float(entry["minimum"]), current_min)
                )
                entry["maximum"] = (
                    current_max
                    if entry["maximum"] is None
                    else max(float(entry["maximum"]), current_max)
                )
            if len(numeric):
                changed = numeric.ne(numeric.shift()) | numeric.isna() | numeric.shift().isna()
                group_ids = changed.cumsum()
                run_lengths = numeric.groupby(group_ids).size()
                entry["longest_constant_run"] = max(
                    int(entry["longest_constant_run"]),
                    int(run_lengths.max()) if not run_lengths.empty else 0,
                )
    schema_groups: dict[str, list[str]] = {}
    for profile in profiles:
        key = json.dumps(profile["column_names"], ensure_ascii=False)
        schema_groups.setdefault(key, []).append(str(profile["file"]))
    result = {
        "source_dir": str(SOURCE_DIR),
        "file_count": len(files),
        "schema_group_count": len(schema_groups),
        "schema_groups": list(schema_groups.values()),
        "profiles": profiles,
        "signal_stats": list(signal_stats.values()),
    }
    OUTPUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote read-only profile: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
