"""Focused completeness audit for the newly exported 260617 process."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline import canonical_signal
from protocol import default_source_dir, project_dir


def main() -> None:
    source = next(default_source_dir().glob("Original__*260617-ALL.csv"))
    frame = pd.read_csv(source, low_memory=False)
    time_columns = [column for column in frame if column.endswith(" Time")]
    value_columns = [column for column in frame if column.endswith(" ValueY")]
    reference_time = frame[time_columns[0]].astype(str)
    time_mismatch_cells = sum(
        int((frame[column].astype(str) != reference_time).sum()) for column in time_columns[1:]
    )
    timestamp = pd.to_datetime(reference_time, format="%Y/%m/%d %H:%M:%S", errors="coerce")
    delta = timestamp.diff().dt.total_seconds().dropna().to_numpy()
    values = frame[value_columns].apply(pd.to_numeric, errors="coerce")
    signal_rows = []
    for column in value_columns:
        signal = canonical_signal(column[: -len(" ValueY")])
        series = values[column].to_numpy(dtype=float)
        finite = series[np.isfinite(series)]
        signal_rows.append({
            "signal": signal,
            "rows": len(series),
            "missing": int((~np.isfinite(series)).sum()),
            "unique_finite": int(pd.Series(finite).nunique()),
            "minimum": float(np.min(finite)),
            "maximum": float(np.max(finite)),
            "std": float(np.std(finite)),
            "constant_in_0617": bool(np.ptp(finite) == 0.0),
        })
    signal_frame = pd.DataFrame(signal_rows)
    report = {
        "status": "PASS" if (
            len(frame) == 22680
            and len(time_columns) == 54
            and len(value_columns) == 54
            and time_mismatch_cells == 0
            and int(values.isna().sum().sum()) == 0
            and int((delta != 10.0).sum()) == 0
        ) else "FAIL",
        "file": source.name,
        "rows": len(frame),
        "columns": len(frame.columns),
        "signal_pairs": len(value_columns),
        "start_time": str(timestamp.iloc[0]),
        "end_time": str(timestamp.iloc[-1]),
        "duplicate_timestamps": int(timestamp.duplicated().sum()),
        "irregular_10s_intervals": int((delta != 10.0).sum()),
        "invalid_timestamps": int(timestamp.isna().sum()),
        "cross_signal_timestamp_mismatch_cells": time_mismatch_cells,
        "missing_numeric_cells": int(values.isna().sum().sum()),
        "rows_with_missing_numeric": int(values.isna().any(axis=1).sum()),
        "constant_signal_count_in_0617": int(signal_frame["constant_in_0617"].sum()),
        "constant_signals_in_0617": signal_frame.loc[
            signal_frame["constant_in_0617"], "signal"
        ].tolist(),
    }
    output = project_dir() / "reports"
    output.mkdir(parents=True, exist_ok=True)
    signal_frame.to_csv(output / "audit_0617_signal_summary.csv", index=False)
    (output / "audit_0617.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
