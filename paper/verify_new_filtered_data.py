#!/usr/bin/env python3
"""Read-only verification of the nine filtered HTC 8300 CSV files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


MERGED_DIR = Path(r"C:\Users\Administrator\Desktop\新建文件夹\csv")
FILTERED_DIR = Path(r"C:\Users\Administrator\Desktop\新建文件夹\filtered")
OUTPUT = Path(__file__).resolve().parent / "new_filtered_readonly_verification.json"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def main() -> None:
    merged_files = sorted(MERGED_DIR.glob("HTC*.csv"))
    filtered_files = sorted(FILTERED_DIR.glob("HTC*.csv"))
    if [path.name for path in merged_files] != [path.name for path in filtered_files]:
        raise AssertionError("Merged and filtered file sets differ")
    file_rows = []
    blank_counts: dict[str, int] = {}
    all_blank_in_every_file: set[str] | None = None
    for merged_path, filtered_path in zip(merged_files, filtered_files):
        merged = read_csv(merged_path)
        filtered = read_csv(filtered_path)
        if list(merged.columns) != list(filtered.columns) or len(merged) != len(filtered):
            raise AssertionError(f"Shape/schema differs: {filtered_path.name}")
        time_columns = [column for column in merged.columns if column.endswith(" Time")]
        for column in time_columns:
            if not merged[column].fillna("<NA>").astype(str).equals(
                filtered[column].fillna("<NA>").astype(str)
            ):
                raise AssertionError(f"Time column changed: {filtered_path.name} / {column}")
        value_columns = [column for column in filtered.columns if column.endswith(" ValueY")]
        all_blank_here = set()
        blanks = 0
        for column in value_columns:
            numeric = pd.to_numeric(filtered[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            count = int(numeric.isna().sum())
            blanks += count
            blank_counts[column] = blank_counts.get(column, 0) + count
            if count == len(filtered):
                all_blank_here.add(column[: -len(" ValueY")])
        all_blank_in_every_file = (
            all_blank_here
            if all_blank_in_every_file is None
            else all_blank_in_every_file & all_blank_here
        )
        file_rows.append(
            {
                "file": filtered_path.name,
                "rows": len(filtered),
                "columns": len(filtered.columns),
                "blank_numeric_cells": blanks,
            }
        )
    result = {
        "file_count": len(filtered_files),
        "schema_and_row_counts_match_merged": True,
        "time_columns_unchanged": True,
        "files": file_rows,
        "total_blank_numeric_cells": sum(blank_counts.values()),
        "all_blank_signals_in_every_filtered_file": sorted(all_blank_in_every_file or set()),
        "all_blank_signal_count": len(all_blank_in_every_file or set()),
        "other_columns_with_blanks": {
            column[: -len(" ValueY")]: count
            for column, count in sorted(blank_counts.items())
            if count > 0
            and column[: -len(" ValueY")] not in (all_blank_in_every_file or set())
        },
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
