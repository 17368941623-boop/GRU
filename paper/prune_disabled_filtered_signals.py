#!/usr/bin/env python3
"""Remove 34 confirmed-disabled signals from the nine filtered CSV files.

Both the Time and ValueY columns are removed for each disabled signal.  The
filter manifest hashes and schema counts are then updated atomically so that
the audited augmentation pipeline can verify the modified parents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DISABLED_SIGNALS = (
    "CV8309",
    "EC-B-T",
    "FC-A-T",
    "FC-B-T",
    "FC-C-T",
    "FC-E-T",
    "FC-G-T",
    "FR8351",
    "HV1-T1",
    "HV1-T2",
    "HV2-T1",
    "HV2-T2",
    "HV3-T1",
    "HV3-T2",
    "HV4-T1",
    "HV4-T2",
    "HV5-T1",
    "HV5-T2",
    "HV6-T1",
    "HV6-T2",
    "HV7-T1",
    "HV7-T2",
    "HV8-T1",
    "HV8-T2",
    "LA-T1",
    "LB-T1",
    "LB-T2",
    "LB-T3",
    "LB-T4",
    "LB-T5",
    "LB-T6",
    "LG-T1",
    "LG-T2",
    "PDT8309",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filtered-dir",
        type=Path,
        default=Path(r"C:\Users\Administrator\Desktop\新建文件夹\filtered"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def signal_columns(signal: str) -> tuple[str, str]:
    return f"{signal} Time", f"{signal} ValueY"


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    reloaded = read_csv(temporary)
    if list(reloaded.columns) != list(frame.columns) or len(reloaded) != len(frame):
        temporary.unlink(missing_ok=True)
        raise AssertionError(f"CSV round-trip failed: {path.name}")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    filtered_dir = args.filtered_dir.resolve()
    data_files = sorted(filtered_dir.glob("HTC*.csv"))
    if len(data_files) != 9:
        raise ValueError(f"Expected 9 filtered data CSVs, found {len(data_files)}")

    expected_drop_columns = [
        column for signal in DISABLED_SIGNALS for column in signal_columns(signal)
    ]

    # Preflight every file before modifying any of them.
    for path in data_files:
        frame = read_csv(path)
        present = [column for column in expected_drop_columns if column in frame.columns]
        if present and len(present) != len(expected_drop_columns):
            missing = sorted(set(expected_drop_columns) - set(present))
            raise ValueError(f"Partial disabled-signal schema in {path.name}: {missing}")
        if present:
            for signal in DISABLED_SIGNALS:
                value_column = signal_columns(signal)[1]
                numeric = pd.to_numeric(frame[value_column], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                )
                nonzero = numeric.dropna().abs().gt(1e-12)
                if nonzero.any():
                    raise ValueError(
                        f"Confirmed-disabled signal contains nonzero values: "
                        f"{path.name} / {signal}"
                    )
        elif len(frame.columns) != 108:
            raise ValueError(
                f"Disabled columns are absent but {path.name} has {len(frame.columns)} columns"
            )

    for index, path in enumerate(data_files, start=1):
        frame = read_csv(path)
        if expected_drop_columns[0] in frame.columns:
            frame = frame.drop(columns=expected_drop_columns)
            atomic_write_csv(frame, path)
        if len(frame.columns) != 108:
            raise AssertionError(f"Expected 108 columns after pruning: {path.name}")
        time_columns = [column for column in frame.columns if column.endswith(" Time")]
        value_columns = [column for column in frame.columns if column.endswith(" ValueY")]
        if len(time_columns) != 54 or len(value_columns) != 54:
            raise AssertionError(f"Expected 54 signal pairs after pruning: {path.name}")
        print(f"[{index}/9] {path.name}: 54 enabled signals, 108 columns")

    manifest_path = filtered_dir / "filter_manifest.csv"
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig", low_memory=False)
    if set(manifest["output_file"].astype(str)) != {path.name for path in data_files}:
        raise AssertionError("filter_manifest.csv does not match the nine filtered files")
    manifest = manifest.copy()
    hash_by_file = {path.name: sha256_file(path) for path in data_files}
    manifest["output_sha256"] = manifest["output_file"].map(hash_by_file)
    manifest["columns"] = 108
    if "filtered_column_count" in manifest.columns:
        manifest["filtered_column_count"] = 34
    if "valve_column_count" in manifest.columns:
        manifest["valve_column_count"] = 18
    manifest["disabled_signal_count_removed"] = len(DISABLED_SIGNALS)
    manifest["disabled_columns_removed"] = len(expected_drop_columns)
    temporary_manifest = manifest_path.with_suffix(".csv.tmp")
    manifest.to_csv(temporary_manifest, index=False, encoding="utf-8-sig")
    temporary_manifest.replace(manifest_path)

    for report_name in (
        "temperature_flypoint_events.csv",
        "temperature_flypoint_endpoint_exclusions.csv",
    ):
        report_path = filtered_dir / report_name
        report = pd.read_csv(report_path, encoding="utf-8-sig", low_memory=False)
        if not report.empty:
            disabled_events = report["signal"].astype(str).isin(DISABLED_SIGNALS)
            if disabled_events.any():
                raise AssertionError(
                    f"{report_name} unexpectedly references a removed signal"
                )

    config_path = filtered_dir / "filter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "disabled_signals_removed": list(DISABLED_SIGNALS),
            "disabled_signal_count_removed": len(DISABLED_SIGNALS),
            "disabled_columns_removed": len(expected_drop_columns),
            "remaining_signal_pairs": 54,
            "columns_per_file_after_disabled_signal_removal": 108,
            "disabled_signal_policy": (
                "confirmed unused by operator; both Time and ValueY columns removed "
                "after filtering and before augmentation"
            ),
            "fc_v1_semantics": "confirmed 0-100 percent valve opening",
        }
    )
    temporary_config = config_path.with_suffix(".json.tmp")
    temporary_config.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_config.replace(config_path)

    removal_record = {
        "filtered_directory": str(filtered_dir),
        "files_modified": len(data_files),
        "signals_removed": list(DISABLED_SIGNALS),
        "removed_signal_count": len(DISABLED_SIGNALS),
        "removed_columns_per_file": len(expected_drop_columns),
        "remaining_signal_pairs": 54,
        "remaining_columns": 108,
        "filter_manifest_hashes_updated": True,
    }
    (filtered_dir / "disabled_signal_removal.json").write_text(
        json.dumps(removal_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Disabled-signal removal and manifest update complete.")


if __name__ == "__main__":
    main()
