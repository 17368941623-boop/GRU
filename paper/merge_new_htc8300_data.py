#!/usr/bin/env python3
"""Merge 16 HTC 8300 historian exports into nine complete cooldown CSV files.

The source files are never modified.  The script verifies schema identity,
10-second timestamp continuity, duplicate timestamps, and missing/constant
signals before writing merged files and audit reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


GROUPS: dict[str, tuple[str, ...]] = {
    "HTC8300趋势图-251130-ALL.csv": (
        "HTC 8300趋势图 1130 8-16 88.csv",
        "HTC 8300趋势图 1130 16-23 88.csv",
    ),
    "HTC8300趋势图-251226-ALL.csv": (
        "HTC 8300趋势图 1226 9-21 88.csv",
        "HTC 8300趋势图 1226 21-20 88.csv",
    ),
    "HTC8300趋势图-260118-ALL.csv": (
        "HTC 8300趋势图 0118 12-19 88.csv",
        "HTC 8300趋势图 0118 19-16 88.csv",
    ),
    "HTC8300趋势图-260403-ALL.csv": (
        "HTC 8300趋势图 0403 9-15 88.csv",
        "HTC 8300趋势图 0403 15-21 88.csv",
    ),
    "HTC8300趋势图-260428-ALL.csv": (
        "HTC 8300趋势图 0428 10-12 88.csv",
        "HTC 8300趋势图 0428 12-0 88.csv",
    ),
    "HTC8300趋势图-260617-ALL.csv": (
        "HTC 8300趋势图 0617 8-16 88.csv",
        "HTC 8300趋势图 0617 16-23 88.csv",
    ),
    "HTC8300趋势图-260623-BACK-ALL.csv": (
        "HTC 8300趋势图 0623BACK 20-16 88.csv",
    ),
    "HTC8300趋势图-260715-ALL.csv": (
        "HTC 8300趋势图 0715 10-15 88.csv",
        "HTC 8300趋势图 0715 15-16 88.csv",
    ),
    "HTC8300趋势图-260721-BACK-ALL.csv": (
        "HTC 8300趋势图 0721BACK  88.csv",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(r"C:\Users\Administrator\Desktop\新建文件夹"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\Users\Administrator\Desktop\新建文件夹\csv"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
    raise ValueError(f"Unable to read {path}: {' | '.join(errors)}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def longest_true_run(mask: pd.Series) -> int:
    values = mask.to_numpy(dtype=bool, copy=False)
    if values.size == 0 or not values.any():
        return 0
    padded = np.r_[False, values, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return int((edges[1::2] - edges[::2]).max())


def longest_constant_run(series: pd.Series) -> int:
    if series.empty:
        return 0
    changed = series.ne(series.shift()) | series.isna() | series.shift().isna()
    run_lengths = series.groupby(changed.cumsum()).size()
    return int(run_lengths.max()) if not run_lengths.empty else 0


def verify_time_columns(frame: pd.DataFrame, file_name: str) -> tuple[str, pd.Series, float]:
    time_columns = [column for column in frame.columns if column.endswith(" Time")]
    value_columns = [column for column in frame.columns if column.endswith(" ValueY")]
    if not time_columns or len(time_columns) != len(value_columns):
        raise ValueError(
            f"{file_name}: expected paired Time/ValueY columns, found "
            f"{len(time_columns)} time and {len(value_columns)} value columns"
        )
    primary = time_columns[0]
    for column in time_columns[1:]:
        if not frame[column].fillna("<NA>").astype(str).equals(
            frame[primary].fillna("<NA>").astype(str)
        ):
            raise ValueError(f"{file_name}: time column {column} differs from {primary}")
    parsed = pd.to_datetime(frame[primary], errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{file_name}: {int(parsed.isna().sum())} invalid timestamps")
    diffs = parsed.diff().dt.total_seconds().dropna()
    if diffs.empty:
        raise ValueError(f"{file_name}: fewer than two timestamps")
    positive = diffs[diffs > 0]
    if positive.empty:
        raise ValueError(f"{file_name}: no increasing timestamps")
    median_dt = float(positive.median())
    return primary, parsed, median_dt


def validate_source_set(source_dir: Path) -> list[Path]:
    expected = {name for group in GROUPS.values() for name in group}
    actual = {path.name for path in source_dir.glob("*.csv")}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            "Source file set differs from the expected 16 exports. "
            f"Missing={missing}; unexpected={unexpected}"
        )
    return [source_dir / name for name in sorted(expected)]


def write_csv(rows: Iterable[dict[str, object]], path: Path, columns: list[str]) -> None:
    pd.DataFrame(list(rows), columns=columns).to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if source_dir == output_dir:
        raise ValueError("output-dir must differ from source-dir")
    source_files = validate_source_set(source_dir)
    source_hashes_before = {path.name: sha256_file(path) for path in source_files}

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [output_dir / name for name in GROUPS]
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{len(existing)} merged files already exist. Use --overwrite to replace them."
        )

    segment_rows: list[dict[str, object]] = []
    process_rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    reference_columns: list[str] | None = None

    for process_index, (output_name, source_names) in enumerate(GROUPS.items(), start=1):
        pieces: list[tuple[Path, pd.DataFrame, pd.Series, float]] = []
        for source_name in source_names:
            source_path = source_dir / source_name
            frame = read_export(source_path)
            if reference_columns is None:
                reference_columns = list(frame.columns)
            elif list(frame.columns) != reference_columns:
                raise ValueError(f"Schema/order mismatch in {source_name}")
            primary, timestamps, median_dt = verify_time_columns(frame, source_name)
            pieces.append((source_path, frame, timestamps, median_dt))

        pieces.sort(key=lambda item: item[2].iloc[0])
        for segment_index, (source_path, frame, timestamps, median_dt) in enumerate(pieces, start=1):
            diffs = timestamps.diff().dt.total_seconds()
            irregular = diffs.notna() & ~np.isclose(diffs, median_dt, rtol=0.0, atol=1e-9)
            segment_rows.append(
                {
                    "process_file": output_name,
                    "segment_order": segment_index,
                    "source_file": source_path.name,
                    "rows": len(frame),
                    "columns": len(frame.columns),
                    "start_time": timestamps.iloc[0],
                    "end_time": timestamps.iloc[-1],
                    "median_interval_seconds": median_dt,
                    "duplicate_timestamps": int(timestamps.duplicated(keep=False).sum()),
                    "irregular_intervals": int(irregular.sum()),
                    "source_sha256": source_hashes_before[source_path.name],
                }
            )

        for previous, current in zip(pieces, pieces[1:]):
            previous_end = previous[2].iloc[-1]
            current_start = current[2].iloc[0]
            expected_seconds = previous[3]
            actual_seconds = float((current_start - previous_end).total_seconds())
            if not np.isclose(actual_seconds, expected_seconds, rtol=0.0, atol=1e-9):
                gap_rows.append(
                    {
                        "process_file": output_name,
                        "before_file": previous[0].name,
                        "after_file": current[0].name,
                        "before_end": previous_end,
                        "after_start": current_start,
                        "expected_interval_seconds": expected_seconds,
                        "actual_interval_seconds": actual_seconds,
                        "missing_intervals": max(
                            int(round(actual_seconds / expected_seconds)) - 1, 0
                        ),
                    }
                )
                raise ValueError(
                    f"Boundary gap/overlap in {output_name}: {previous[0].name} -> "
                    f"{current[0].name}, {actual_seconds}s"
                )

        merged = pd.concat([item[1] for item in pieces], ignore_index=True)
        primary, timestamps, median_dt = verify_time_columns(merged, output_name)
        diffs = timestamps.diff().dt.total_seconds()
        irregular_mask = diffs.notna() & ~np.isclose(diffs, median_dt, rtol=0.0, atol=1e-9)
        for row_index in np.flatnonzero(irregular_mask.to_numpy()):
            gap_rows.append(
                {
                    "process_file": output_name,
                    "before_file": "merged",
                    "after_file": "merged",
                    "before_end": timestamps.iloc[row_index - 1],
                    "after_start": timestamps.iloc[row_index],
                    "expected_interval_seconds": median_dt,
                    "actual_interval_seconds": diffs.iloc[row_index],
                    "missing_intervals": max(
                        int(round(float(diffs.iloc[row_index]) / median_dt)) - 1, 0
                    ),
                }
            )
        if timestamps.duplicated().any() or irregular_mask.any():
            raise ValueError(f"Merged timestamp validation failed for {output_name}")

        destination = output_dir / output_name
        merged.to_csv(destination, index=False, encoding="utf-8-sig")
        reloaded = read_export(destination)
        _, reloaded_time, _ = verify_time_columns(reloaded, output_name)
        if list(reloaded.columns) != list(merged.columns) or len(reloaded) != len(merged):
            destination.unlink(missing_ok=True)
            raise AssertionError(f"CSV round-trip validation failed for {output_name}")
        if not timestamps.reset_index(drop=True).equals(reloaded_time.reset_index(drop=True)):
            destination.unlink(missing_ok=True)
            raise AssertionError(f"Timestamp round-trip validation failed for {output_name}")

        missing_cells = 0
        fully_missing_columns = 0
        all_zero_columns = 0
        constant_columns = 0
        for column in [name for name in merged.columns if name.endswith(" ValueY")]:
            numeric = pd.to_numeric(merged[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            missing_count = int(numeric.isna().sum())
            zero_count = int(numeric.eq(0.0).sum())
            unique_nonmissing = int(numeric.nunique(dropna=True))
            is_fully_missing = missing_count == len(numeric)
            is_all_zero = zero_count == len(numeric)
            is_constant = unique_nonmissing <= 1
            missing_cells += missing_count
            fully_missing_columns += int(is_fully_missing)
            all_zero_columns += int(is_all_zero)
            constant_columns += int(is_constant)
            signal_rows.append(
                {
                    "process_file": output_name,
                    "signal": column[: -len(" ValueY")],
                    "column": column,
                    "rows": len(numeric),
                    "missing_count": missing_count,
                    "missing_fraction": missing_count / max(len(numeric), 1),
                    "longest_missing_run_rows": longest_true_run(numeric.isna()),
                    "zero_count": zero_count,
                    "zero_fraction": zero_count / max(len(numeric), 1),
                    "longest_zero_run_rows": longest_true_run(numeric.eq(0.0)),
                    "negative_count": int(numeric.lt(0.0).sum()),
                    "unique_nonmissing_values": unique_nonmissing,
                    "longest_constant_run_rows": longest_constant_run(numeric),
                    "minimum": float(numeric.min()) if numeric.notna().any() else np.nan,
                    "maximum": float(numeric.max()) if numeric.notna().any() else np.nan,
                    "fully_missing": is_fully_missing,
                    "all_zero": is_all_zero,
                    "constant": is_constant,
                }
            )

        process_rows.append(
            {
                "process_file": output_name,
                "source_segment_count": len(pieces),
                "rows": len(merged),
                "columns": len(merged.columns),
                "signal_pairs": sum(name.endswith(" ValueY") for name in merged.columns),
                "start_time": timestamps.iloc[0],
                "end_time": timestamps.iloc[-1],
                "duration_hours": (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / 3600,
                "sample_interval_seconds": median_dt,
                "duplicate_timestamps": int(timestamps.duplicated(keep=False).sum()),
                "irregular_intervals": int(irregular_mask.sum()),
                "missing_numeric_cells": missing_cells,
                "fully_missing_signal_columns": fully_missing_columns,
                "all_zero_signal_columns": all_zero_columns,
                "constant_signal_columns": constant_columns,
                "output_sha256": sha256_file(destination),
            }
        )
        print(f"[{process_index}/{len(GROUPS)}] wrote {destination.name}: {len(merged)} rows")

    source_hashes_after = {path.name: sha256_file(path) for path in source_files}
    if source_hashes_before != source_hashes_after:
        raise AssertionError("One or more source files changed unexpectedly")

    write_csv(
        segment_rows,
        audit_dir / "segment_merge_manifest.csv",
        [
            "process_file",
            "segment_order",
            "source_file",
            "rows",
            "columns",
            "start_time",
            "end_time",
            "median_interval_seconds",
            "duplicate_timestamps",
            "irregular_intervals",
            "source_sha256",
        ],
    )
    write_csv(
        process_rows,
        audit_dir / "process_quality_summary.csv",
        [
            "process_file",
            "source_segment_count",
            "rows",
            "columns",
            "signal_pairs",
            "start_time",
            "end_time",
            "duration_hours",
            "sample_interval_seconds",
            "duplicate_timestamps",
            "irregular_intervals",
            "missing_numeric_cells",
            "fully_missing_signal_columns",
            "all_zero_signal_columns",
            "constant_signal_columns",
            "output_sha256",
        ],
    )
    write_csv(
        signal_rows,
        audit_dir / "signal_quality_summary.csv",
        [
            "process_file",
            "signal",
            "column",
            "rows",
            "missing_count",
            "missing_fraction",
            "longest_missing_run_rows",
            "zero_count",
            "zero_fraction",
            "longest_zero_run_rows",
            "negative_count",
            "unique_nonmissing_values",
            "longest_constant_run_rows",
            "minimum",
            "maximum",
            "fully_missing",
            "all_zero",
            "constant",
        ],
    )
    write_csv(
        gap_rows,
        audit_dir / "time_gap_report.csv",
        [
            "process_file",
            "before_file",
            "after_file",
            "before_end",
            "after_start",
            "expected_interval_seconds",
            "actual_interval_seconds",
            "missing_intervals",
        ],
    )

    signal_frame = pd.DataFrame(signal_rows)
    global_all_zero = sorted(
        signal
        for signal, group in signal_frame.groupby("signal", sort=True)
        if bool(group["all_zero"].all())
    )
    global_constant = sorted(
        signal
        for signal, group in signal_frame.groupby("signal", sort=True)
        if bool(group["constant"].all())
    )
    summary = {
        "source_directory": str(source_dir),
        "merged_directory": str(output_dir),
        "source_files": len(source_files),
        "merged_processes": len(process_rows),
        "schema_groups": 1,
        "columns_per_file": int(process_rows[0]["columns"]),
        "signal_pairs_per_file": int(process_rows[0]["signal_pairs"]),
        "sample_interval_seconds": 10.0,
        "total_rows": int(sum(int(row["rows"]) for row in process_rows)),
        "time_gaps_or_overlaps": len(gap_rows),
        "duplicate_timestamps": int(
            sum(int(row["duplicate_timestamps"]) for row in process_rows)
        ),
        "missing_numeric_cells": int(
            sum(int(row["missing_numeric_cells"]) for row in process_rows)
        ),
        "globally_all_zero_signals": global_all_zero,
        "globally_all_zero_signal_count": len(global_all_zero),
        "globally_constant_signals": global_constant,
        "globally_constant_signal_count": len(global_constant),
        "source_files_unchanged": True,
    }
    (audit_dir / "dataset_quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Merge and audit complete: {output_dir}")


if __name__ == "__main__":
    main()
