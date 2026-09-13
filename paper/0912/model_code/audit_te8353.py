"""Audit TE8353 presence and compare overlapping legacy/new timestamps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\Administrator\Desktop\论文20260708")
LEGACY = ROOT / "0822" / "original_data"
NEW_ROOT = Path(r"C:\Users\Administrator\Desktop\新建文件夹")
REPORT_DIR = ROOT / "0912" / "reports"

PAIRS = (
    ("HTC8300趋势图-251130.csv", "HTC8300趋势图-251130-ALL.csv"),
    ("HTC8300趋势图-251226.csv", "HTC8300趋势图-251226-ALL.csv"),
    ("HTC8300趋势图-260118.csv", "HTC8300趋势图-260118-ALL.csv"),
    ("HTC8300趋势图-260403.csv", "HTC8300趋势图-260403-ALL.csv"),
    ("HTC 8300趋势图-0617-ALL.csv", "HTC8300趋势图-260617-ALL.csv"),
    ("HTC 8300趋势图-0623-ALL-BACK045.csv", "HTC8300趋势图-260623-BACK-ALL.csv"),
    ("HTC 8300趋势图-0715-ALL.csv", "HTC8300趋势图-260715-ALL.csv"),
)


def detect_encoding(path: Path) -> str:
    """Return a practical encoding for exported historian CSV files."""
    prefix = path.read_bytes()[:4]
    if prefix.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    if prefix.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    for encoding in ("utf-8", "gb18030"):
        try:
            path.read_text(encoding=encoding, errors="strict")
            return encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Cannot determine CSV encoding: {path}")


def csv_read_options(path: Path) -> dict[str, str]:
    encoding = detect_encoding(path)
    with path.open("r", encoding=encoding, newline="") as handle:
        first_line = handle.readline()
    separator = "\t" if first_line.count("\t") > first_line.count(",") else ","
    return {"encoding": encoding, "sep": separator}


def read_te8353(path: Path) -> pd.DataFrame:
    options = csv_read_options(path)
    header = pd.read_csv(path, nrows=0, **options)
    required = ["TE8353 Time", "TE8353 ValueY"]
    missing = [column for column in required if column not in header.columns]
    if missing:
        raise KeyError(f"{path}: missing {missing}")
    frame = pd.read_csv(path, usecols=required, low_memory=False, **options)
    frame["timestamp"] = pd.to_datetime(frame["TE8353 Time"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["TE8353 ValueY"], errors="coerce")
    return frame[["timestamp", "value"]]


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    presence_rows = []
    for stage, directory, pattern in (
        ("merged_raw", NEW_ROOT / "csv", "HTC*.csv"),
        ("filtered", NEW_ROOT / "filtered", "HTC*.csv"),
        ("augmented_original", NEW_ROOT / "filtered_data", "Original__HTC*.csv"),
    ):
        for path in sorted(directory.glob(pattern)):
            header = pd.read_csv(path, nrows=0, **csv_read_options(path))
            has_time = "TE8353 Time" in header.columns
            has_value = "TE8353 ValueY" in header.columns
            row = {
                "stage": stage,
                "file": path.name,
                "has_TE8353_Time": has_time,
                "has_TE8353_ValueY": has_value,
            }
            if has_time and has_value:
                values = read_te8353(path)
                finite = values["value"].to_numpy(dtype=float)
                row.update({
                    "rows": len(values),
                    "missing_timestamps": int(values["timestamp"].isna().sum()),
                    "missing_values": int((~np.isfinite(finite)).sum()),
                    "minimum_k": float(np.nanmin(finite)),
                    "maximum_k": float(np.nanmax(finite)),
                    "std_k": float(np.nanstd(finite)),
                })
            presence_rows.append(row)

    comparison_rows = []
    for legacy_name, new_name in PAIRS:
        legacy_path = LEGACY / legacy_name
        new_raw_path = NEW_ROOT / "csv" / new_name
        new_filtered_path = NEW_ROOT / "filtered_data" / f"Original__{new_name}"
        legacy = read_te8353(legacy_path).rename(columns={"value": "legacy_value"})
        legacy["minute"] = legacy["timestamp"].dt.floor("min")
        legacy["minute_order"] = legacy.groupby("minute", dropna=False).cumcount()
        for stage, new_path in (("merged_raw", new_raw_path), ("filtered", new_filtered_path)):
            current = read_te8353(new_path).rename(columns={"value": "new_value"})
            current["minute"] = current["timestamp"].dt.floor("min")
            current["minute_order"] = current.groupby("minute", dropna=False).cumcount()
            merged = legacy.merge(
                current,
                on=["minute", "minute_order"],
                how="inner",
                suffixes=("_legacy", "_new"),
            )
            difference = np.abs(
                merged["legacy_value"].to_numpy(dtype=float)
                - merged["new_value"].to_numpy(dtype=float)
            )
            comparison_rows.append({
                "legacy_file": legacy_name,
                "new_file": new_path.name,
                "new_stage": stage,
                "legacy_rows": len(legacy),
                "new_rows": len(current),
                "overlap_minute_positions": len(merged),
                "exact_equal_values": int((difference == 0.0).sum()),
                "values_with_abs_diff_gt_1e_9": int((difference > 1e-9).sum()),
                "mean_abs_difference_k": float(np.mean(difference)) if len(difference) else None,
                "max_abs_difference_k": float(np.max(difference)) if len(difference) else None,
            })

    presence = pd.DataFrame(presence_rows)
    comparisons = pd.DataFrame(comparison_rows)
    presence.to_csv(REPORT_DIR / "te8353_presence_audit.csv", index=False)
    comparisons.to_csv(REPORT_DIR / "te8353_legacy_overlap_comparison.csv", index=False)
    report = {
        "status": "PASS" if (
            presence["has_TE8353_Time"].all()
            and presence["has_TE8353_ValueY"].all()
            and int(presence["missing_values"].fillna(0).sum()) == 0
        ) else "FAIL",
        "new_files_checked": int(len(presence)),
        "new_files_missing_te8353_column": int(
            (~(presence["has_TE8353_Time"] & presence["has_TE8353_ValueY"])).sum()
        ),
        "new_te8353_missing_values": int(presence["missing_values"].fillna(0).sum()),
        "legacy_new_process_pairs_compared": len(PAIRS),
        "decision": "No backfill is needed when status is PASS; preserve the newer complete exports.",
    }
    (REPORT_DIR / "te8353_audit_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
