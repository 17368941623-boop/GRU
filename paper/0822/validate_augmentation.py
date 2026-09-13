"""Verify that augmentation changed only the intended temperature values."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import create as creat


OUTPUT_DIR = creat.DEFAULT_OUTPUT_DIR
REPORT_PATH = OUTPUT_DIR / "augmentation_validation.csv"


def load_output(path: Path) -> pd.DataFrame:
    return creat.normalize_columns(pd.read_csv(path, encoding="utf-8", low_memory=False))


def numeric_difference(left: pd.Series, right: pd.Series) -> tuple[bool, float]:
    left_numeric = pd.to_numeric(left, errors="coerce")
    right_numeric = pd.to_numeric(right, errors="coerce")
    if not left_numeric.notna().any() and not right_numeric.notna().any():
        same = left.fillna("<NA>").astype(str).equals(right.fillna("<NA>").astype(str))
        return same, 0.0 if same else float("inf")
    if not left_numeric.isna().equals(right_numeric.isna()):
        return False, float("inf")
    valid = left_numeric.notna()
    if not valid.any():
        return True, 0.0
    maximum = float(
        np.max(np.abs(left_numeric.loc[valid].to_numpy() - right_numeric.loc[valid].to_numpy()))
    )
    return maximum <= 1e-12, maximum


def main() -> None:
    manifest = pd.read_csv(OUTPUT_DIR / "augmentation_manifest.csv", encoding="utf-8-sig")
    rows: list[dict[str, object]] = []
    source_files = creat.discover_filtered_parents(creat.DEFAULT_INPUT_DIR)
    for source_path in source_files:
        source, _, _ = creat.read_industrial_csv(source_path)
        original_path = OUTPUT_DIR / f"Original__{source_path.name}"
        original = load_output(original_path)
        if source.shape != original.shape or list(source.columns) != list(original.columns):
            raise AssertionError(f"Original output schema differs from source: {source_path.name}")
        for column in source.columns:
            equal, difference = numeric_difference(source[column], original[column])
            if not equal:
                raise AssertionError(
                    f"Original output differs from source in {source_path.name}/{column}: {difference}"
                )

        temperature_columns = [
            column for column in original.columns if creat.is_augmented_temperature_column(column)
        ]
        non_temperature_columns = [
            column for column in original.columns if column not in temperature_columns
        ]
        thv_column = next(
            column
            for column in temperature_columns
            if creat.signal_name(column).upper() == "THV"
        )
        for variant_id in (1, 2):
            augmented_path = OUTPUT_DIR / f"Aug_{variant_id:02d}__{source_path.name}"
            augmented = load_output(augmented_path)
            schema_ok = original.shape == augmented.shape and list(original.columns) == list(
                augmented.columns
            )
            if not schema_ok:
                raise AssertionError(f"Augmented schema differs: {augmented_path.name}")

            non_temperature_max_difference = 0.0
            non_temperature_equal = True
            for column in non_temperature_columns:
                equal, difference = numeric_difference(original[column], augmented[column])
                non_temperature_equal &= equal
                non_temperature_max_difference = max(
                    non_temperature_max_difference, difference
                )

            target_statistics: dict[str, float | bool] = {}
            masks_ok = True
            for label, column in (("te8353", "TE8353 ValueY"), ("thv", thv_column)):
                before = pd.to_numeric(original[column], errors="coerce")
                after = pd.to_numeric(augmented[column], errors="coerce")
                masks_ok &= before.isna().equals(after.isna())
                masks_ok &= before.eq(0.0).equals(after.eq(0.0))
                valid = before.notna() & before.ne(0.0)
                absolute = (after.loc[valid] - before.loc[valid]).abs()
                target_statistics[f"{label}_changed"] = bool(absolute.gt(1e-12).any())
                target_statistics[f"{label}_mean_abs_change_k"] = float(absolute.mean())
                target_statistics[f"{label}_max_abs_change_k"] = float(absolute.max())

            ab_columns = [column for column in original.columns if creat.signal_name(column) in {"A管", "B管"}]
            ab_equal = True
            for column in ab_columns:
                equal, _ = numeric_difference(original[column], augmented[column])
                ab_equal &= equal

            status = bool(
                schema_ok
                and non_temperature_equal
                and ab_equal
                and masks_ok
                and target_statistics["te8353_changed"]
                and target_statistics["thv_changed"]
            )
            rows.append(
                {
                    "parent_file": source_path.name,
                    "variant_id": variant_id,
                    "rows": len(augmented),
                    "columns": len(augmented.columns),
                    "temperature_columns": len(temperature_columns),
                    "schema_ok": schema_ok,
                    "non_temperature_equal": non_temperature_equal,
                    "non_temperature_max_difference": non_temperature_max_difference,
                    "a_b_pipe_equal": ab_equal,
                    "zero_and_missing_masks_preserved": masks_ok,
                    **target_statistics,
                    "status": "PASS" if status else "FAIL",
                }
            )

    report = pd.DataFrame(rows)
    report.to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")
    expected_manifest_rows = len(source_files) * (creat.NUM_AUGMENTED_VARIANTS + 1)
    expected_report_rows = len(source_files) * creat.NUM_AUGMENTED_VARIANTS
    if len(manifest) != expected_manifest_rows:
        raise AssertionError(
            f"Expected {expected_manifest_rows} manifest rows, found {len(manifest)}"
        )
    if len(report) != expected_report_rows or not report["status"].eq("PASS").all():
        raise AssertionError("Augmentation validation failed")
    data_files = list(OUTPUT_DIR.glob("Original__*.csv")) + list(
        OUTPUT_DIR.glob("Aug_*.csv")
    )
    print(
        report[
            [
                "parent_file",
                "variant_id",
                "te8353_mean_abs_change_k",
                "thv_mean_abs_change_k",
                "status",
            ]
        ].to_string(index=False)
    )
    print(
        f"Validated {len(data_files)} data CSV files; "
        f"report rows={len(report)}; all PASS.\nReport={REPORT_PATH}"
    )


if __name__ == "__main__":
    main()
