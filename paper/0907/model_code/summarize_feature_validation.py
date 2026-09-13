#!/usr/bin/env python3
"""Freeze validation-only results for all 0907 physical-feature ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_protocol import (
    FEATURE_SPECS, FEATURE_GROUPS, MODEL_NAME, SELECTION_METRIC, all_tasks,
    protocol_payload, run_directory, valid_completed_run,
)
from statistics_helpers import interval, paired_table

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_DIR / "outputs_0907"
METRICS = (
    "validation_rmse_k", "validation_mae_k", "validation_p95_absolute_error_k",
    "validation_max_absolute_error_k", "validation_bias_k", "validation_r2",
    "validation_direction_accuracy",
)


def collect(results_dir: Path) -> pd.DataFrame:
    rows, missing = [], []
    for spec, seed in all_tasks():
        if not valid_completed_run(results_dir, spec, seed):
            missing.append(f"{spec['feature_set']}/seed_{seed}")
            continue
        path = run_directory(results_dir, spec, seed) / "metrics.json"
        p = json.loads(path.read_text(encoding="utf-8"))
        v = p["validation_metrics"]
        rows.append({
            "model": MODEL_NAME, "feature_set": spec["feature_set"],
            "feature_set_label": spec["label"], "feature_set_role": spec["role"],
            "removed_feature_group": spec.get("removed_group"), "seed": seed,
            "active_engineered_count": len(spec["engineered"]),
            "trainable_parameters": int(p["trainable_parameters"]),
            "completed_epochs": int(p["completed_epochs"]),
            "best_epoch": int(p["best_epoch"]),
            "training_seconds_total": float(p["training_seconds_total"]),
            "validation_rmse_k": float(v["rmse_k"]),
            "validation_mae_k": float(v["mae_k"]),
            "validation_p95_absolute_error_k": float(v["p95_absolute_error_k"]),
            "validation_max_absolute_error_k": float(v["max_absolute_error_k"]),
            "validation_bias_k": float(v["bias_k"]),
            "validation_r2": float(v["r2"]),
            "validation_direction_accuracy": float(p["validation_direction_accuracy"]),
            "validation_windows": int(p["validation_windows"]),
            "persistence_validation_rmse_k": float(p["persistence_validation_metrics"]["rmse_k"]),
            "checkpoint_selection_metric": p["checkpoint_selection_metric"],
            "test_data_loaded": bool(p["test_data_loaded"]),
            "source_metrics": str(path.relative_to(results_dir)),
        })
    if missing:
        out = results_dir / "validation_summary"
        out.mkdir(parents=True, exist_ok=True)
        (out / "missing_runs.txt").write_text("\n".join(missing)+"\n",encoding="utf-8")
        raise RuntimeError(f"{len(missing)} runs are incomplete")
    data = pd.DataFrame(rows)
    if len(data) != 100 or data.test_data_loaded.any():
        raise ValueError("Training completeness or no-test-access contract failed")
    if data.trainable_parameters.nunique() != 1:
        raise ValueError("Ablations do not have an identical parameter count")
    if set(data.checkpoint_selection_metric) != {SELECTION_METRIC}:
        raise ValueError("Checkpoint selection metric changed")
    return data


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for feature_set, group in runs.groupby("feature_set",sort=False):
        row={
            "feature_set":feature_set,
            "feature_set_label":group.feature_set_label.iloc[0],
            "feature_set_role":group.feature_set_role.iloc[0],
            "removed_feature_group":group.removed_feature_group.iloc[0],
            "n_seeds":group.seed.nunique(),
            "seed_set":";".join(map(str,sorted(group.seed.unique()))),
            "active_engineered_count":int(group.active_engineered_count.iloc[0]),
            "trainable_parameters":int(group.trainable_parameters.iloc[0]),
            "training_minutes_mean":float(group.training_seconds_total.mean()/60),
            "completed_epochs_mean":float(group.completed_epochs.mean()),
        }
        for metric in METRICS:
            values=interval(group[metric].to_numpy(float))
            for key in ("mean","sd","ci95_low","ci95_high"):
                row[f"{metric}_{key}"]=values[key]
        rows.append(row)
    result=pd.DataFrame(rows).sort_values(
        ["validation_rmse_k_mean","validation_rmse_k_sd"]
    ).reset_index(drop=True)
    result.insert(0,"validation_rank",np.arange(1,len(result)+1))
    return result


def comparisons(runs: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    versus_raw = paired_table(runs,[
        (f"{spec['feature_set']}_vs_raw20",str(spec["feature_set"]),"raw20")
        for spec in FEATURE_SPECS if spec["feature_set"] != "raw20"
    ],"validation_rmse_k")
    component = [
        (f"remove_{group}",f"full20_no_{group}","full20")
        for group in FEATURE_GROUPS
    ] + [("add_optional_apparent_heat_leak","full22_with_apparent_heat_leak","full20")]
    return versus_raw, paired_table(runs,component,"validation_rmse_k")


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir",type=Path,default=DEFAULT_RESULTS_DIR)
    args=parser.parse_args(); results=args.results_dir.resolve()
    out=results/"validation_summary"; out.mkdir(parents=True,exist_ok=True)
    runs=collect(results); summary=summarize(runs); vs_raw,component=comparisons(runs)
    runs.to_csv(out/"validation_seed_runs.csv",index=False)
    summary.to_csv(out/"validation_feature_summary.csv",index=False)
    vs_raw.to_csv(out/"paired_vs_raw20.csv",index=False)
    component.to_csv(out/"paired_component_effects.csv",index=False)
    audit={**protocol_payload(),"observed_runs":len(runs),"complete":True,
           "test_data_read_by_training":False,
           "primary_ranking_metric":"complete Original 260501 validation RMSE",
           "uncertainty":"sample SD and two-sided 95% Student-t CI across seeds",
           "paired_test":"two-sided Wilcoxon signed-rank with within-table Holm correction"}
    (out/"validation_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    print(summary.to_string(index=False)); print("VALIDATION_SUMMARY_COMPLETE=true")


if __name__ == "__main__":
    main()
