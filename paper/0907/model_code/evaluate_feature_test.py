#!/usr/bin/env python3
"""Evaluate all frozen 0907 feature-ablation checkpoints on full 0715-BACK."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR=Path(__file__).resolve().parent
PROJECT_DIR=SCRIPT_DIR.parent
SHARED_DIR=PROJECT_DIR/"shared"
if str(SHARED_DIR) not in sys.path: sys.path.insert(0,str(SHARED_DIR))

from train_thv_delta_lstm_lookback import (  # noqa:E402
    FUTURE_CONTROL_COLS, TARGET_COL, Standardizer, choose_device,
    direction_accuracy, make_loader, predict_delta, regression_metrics,
)
import train_thv_delta_rnn_compare as comparison_helpers  # noqa:E402
from train_thv_delta_rnn_compare import (  # noqa:E402
    HorizonControlWindowDataset, assert_horizon_window_boundaries,
    build_horizon_window_ends, label_column, prepare_full_test_frame,
    prepare_horizon_development_frames,
)
from model_components import ModelConfig, build_model  # noqa:E402
from feature_protocol import (  # noqa:E402
    COMMON_ORIGIN_LOOKBACK, FEATURE_GROUPS, FEATURE_SPECS, LOOKBACK,
    MODEL_HISTORY_COLUMNS, MODEL_NAME, PREDICT_STEPS, all_tasks,
    history_feature_mask, protocol_payload, run_directory, valid_completed_run,
)
from statistics_helpers import interval, paired_table  # noqa:E402


def parse_args()->argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,default=PROJECT_DIR/"processed_data")
    p.add_argument("--results-dir",type=Path,default=PROJECT_DIR/"outputs_0907")
    p.add_argument("--batch-size",type=int,default=2048)
    p.add_argument("--num-workers",type=int,default=0)
    p.add_argument("--save-prediction-seeds",default="42")
    p.add_argument("--overwrite",action="store_true")
    return p.parse_args()


def parse_seeds(text:str)->set[int]:
    return {int(item.strip()) for item in text.split(",") if item.strip()}


def load_checkpoint(path:Path)->dict[str,object]:
    try: return torch.load(path,map_location="cpu",weights_only=False)
    except TypeError: return torch.load(path,map_location="cpu")


def standardizer(data:dict[str,object])->Standardizer:
    return Standardizer(mean=np.asarray(data["mean"],dtype=np.float64),
                        scale=np.asarray(data["scale"],dtype=np.float64))


def validate_complete(results:Path)->None:
    missing=[f"{s['feature_set']}/seed_{seed}" for s,seed in all_tasks()
             if not valid_completed_run(results,s,seed)]
    if missing: raise RuntimeError(f"Refusing test access: {len(missing)} runs incomplete")
    if not (results/"validation_summary"/"validation_feature_summary.csv").exists():
        raise FileNotFoundError("Freeze the complete validation summary before test access")


def evaluate_one(args,spec,seed,test_frame,test_ends,current,future,delta,device,save):
    run=run_directory(args.results_dir,spec,seed)
    out=run/"full_test_evaluation"; metric_path=out/"test_metrics.json"
    prediction_path=out/"test_predictions.csv"
    if metric_path.exists() and not args.overwrite and (not save or prediction_path.exists()):
        payload=json.loads(metric_path.read_text(encoding="utf-8")); row=payload["summary_row"]
        if row.get("feature_set")==spec["feature_set"] and int(row.get("seed",-1))==seed:
            print(f"REUSE TEST {spec['feature_set']}/seed_{seed}",flush=True); return row
    checkpoint=load_checkpoint(run/"best_model.pt")
    development=json.loads((run/"metrics.json").read_text(encoding="utf-8"))
    expected={"model_type":MODEL_NAME,"config_id":spec["feature_set"],
              "feature_set":spec["feature_set"],"lookback":LOOKBACK,
              "common_origin_lookback":COMMON_ORIGIN_LOOKBACK,
              "predict_steps":PREDICT_STEPS,"seed":seed,
              "checkpoint_selection_metric":"validation_rmse_k"}
    for key,value in expected.items():
        if checkpoint.get(key)!=value: raise ValueError(f"Checkpoint {key} mismatch: {run}")
    if checkpoint.get("test_data_loaded_during_training",True):
        raise ValueError(f"Checkpoint reports test access: {run}")
    if tuple(checkpoint["history_feature_columns"])!=MODEL_HISTORY_COLUMNS:
        raise ValueError("History feature order mismatch")
    if tuple(checkpoint["future_control_columns"])!=tuple(FUTURE_CONTROL_COLS):
        raise ValueError("Future control order mismatch")
    mask=np.asarray(history_feature_mask(spec),dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(checkpoint["history_feature_mask"]),mask)
    hs=standardizer(checkpoint["history_scaler"])
    cs=standardizer(checkpoint["control_scaler"])
    ts=standardizer(checkpoint["target_scaler"])
    history=hs.transform(test_frame[list(MODEL_HISTORY_COLUMNS)].to_numpy(float)).astype(np.float32)*mask
    controls=cs.transform(test_frame[list(FUTURE_CONTROL_COLS)].to_numpy(float)).astype(np.float32)
    delta_scaled=ts.transform(delta).astype(np.float32)
    dataset=HorizonControlWindowDataset(history,controls,delta_scaled,
        np.ones(len(test_frame),dtype=np.float32),test_ends,LOOKBACK,PREDICT_STEPS)
    loader_args=SimpleNamespace(seed=seed,batch_size=args.batch_size,num_workers=args.num_workers)
    loader=make_loader(dataset,loader_args,shuffle=False,seed_offset=100)
    model=build_model(ModelConfig(**checkpoint["model_config"]),MODEL_HISTORY_COLUMNS,
                      FUTURE_CONTROL_COLS,PREDICT_STEPS,hs.mean,hs.scale)
    model.load_state_dict(checkpoint["model_state_dict"]); model=model.to(device)
    predicted_delta,predicted_ends=predict_delta(model,loader,device,ts)
    if not np.array_equal(predicted_ends,test_ends): raise AssertionError("Test order changed")
    actual=future[test_ends]; current_at_origin=current[test_ends]
    actual_delta=delta[test_ends]; predicted=current_at_origin+predicted_delta
    metrics=regression_metrics(actual,predicted)
    persistence=regression_metrics(actual,current_at_origin)
    row={"model":MODEL_NAME,"feature_set":spec["feature_set"],
      "feature_set_label":spec["label"],"feature_set_role":spec["role"],
      "removed_feature_group":spec.get("removed_group"),"seed":seed,
      "active_engineered_count":len(spec["engineered"]),"lookback":LOOKBACK,
      "common_origin_lookback":COMMON_ORIGIN_LOOKBACK,"predict_steps":PREDICT_STEPS,
      "test_full_rmse_k":float(metrics["rmse_k"]),"test_full_mae_k":float(metrics["mae_k"]),
      "test_full_p95_absolute_error_k":float(metrics["p95_absolute_error_k"]),
      "test_full_max_absolute_error_k":float(metrics["max_absolute_error_k"]),
      "test_full_bias_k":float(metrics["bias_k"]),"test_full_r2":float(metrics["r2"]),
      "test_direction_accuracy":direction_accuracy(actual_delta,predicted_delta),
      "test_windows":int(metrics["n_windows"]),
      "persistence_test_rmse_k":float(persistence["rmse_k"]),
      "validation_rmse_k":float(development["validation_metrics"]["rmse_k"]),
      "trainable_parameters":int(development["trainable_parameters"]),
      "checkpoint_selection_metric":development["checkpoint_selection_metric"]}
    out.mkdir(parents=True,exist_ok=True)
    payload={"evaluation_type":"frozen-checkpoint inference on complete test sequence",
      "training_or_weight_updates":False,"test_parent":"0715-BACK",
      "test_variant":"Original","summary_row":row,"full_test_metrics":metrics,
      "persistence_test_metrics":persistence}
    metric_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    if save:
        pd.DataFrame({"frame_row_index":test_ends,
          "file_id":test_frame.iloc[test_ends].file_id.to_numpy(),
          "source_row_index":test_frame.iloc[test_ends].source_row_index.to_numpy(),
          "source_timestamp":test_frame.iloc[test_ends].source_timestamp.to_numpy(),
          "current_Thv_k":current_at_origin,
          f"actual_delta_Thv_{PREDICT_STEPS}step_k":actual_delta,
          f"predicted_delta_Thv_{PREDICT_STEPS}step_k":predicted_delta,
          f"actual_Future_Thv_{PREDICT_STEPS}step_k":actual,
          f"predicted_Future_Thv_{PREDICT_STEPS}step_k":predicted,
          "residual_k":predicted-actual}).to_csv(prediction_path,index=False)
    print(f"TEST COMPLETE {spec['feature_set']}/seed_{seed} | RMSE={metrics['rmse_k']:.8f}",flush=True)
    return row


def summarize(runs:pd.DataFrame)->pd.DataFrame:
    metric_names=("test_full_rmse_k","test_full_mae_k","test_full_p95_absolute_error_k",
                  "test_full_max_absolute_error_k","test_full_bias_k","test_full_r2",
                  "test_direction_accuracy")
    rows=[]
    for feature_set,g in runs.groupby("feature_set",sort=False):
        row={"feature_set":feature_set,"feature_set_label":g.feature_set_label.iloc[0],
          "feature_set_role":g.feature_set_role.iloc[0],
          "removed_feature_group":g.removed_feature_group.iloc[0],
          "n_seeds":g.seed.nunique(),"seed_set":";".join(map(str,sorted(g.seed.unique()))),
          "active_engineered_count":int(g.active_engineered_count.iloc[0]),
          "trainable_parameters":int(g.trainable_parameters.iloc[0]),
          "test_windows":int(g.test_windows.iloc[0]),
          "persistence_test_rmse_k":float(g.persistence_test_rmse_k.iloc[0])}
        for metric in metric_names:
            values=interval(g[metric].to_numpy(float))
            for key in ("mean","sd","ci95_low","ci95_high"):
                row[f"{metric}_{key}"]=values[key]
        rows.append(row)
    result=pd.DataFrame(rows).sort_values(["test_full_rmse_k_mean","test_full_rmse_k_sd"]).reset_index(drop=True)
    result.insert(0,"test_rank",np.arange(1,len(result)+1)); return result


def main()->None:
    args=parse_args(); args.data_dir=args.data_dir.resolve(); args.results_dir=args.results_dir.resolve()
    validate_complete(args.results_dir); save_seeds=parse_seeds(args.save_prediction_seeds)
    comparison_helpers.HISTORY_FEATURE_COLS=MODEL_HISTORY_COLUMNS
    label=label_column(PREDICT_STEPS)
    train,val=prepare_horizon_development_frames(args.data_dir,PREDICT_STEPS,label)
    test,label_audit=prepare_full_test_frame(args.data_dir,train,val,PREDICT_STEPS,label)
    ends=build_horizon_window_ends(test,COMMON_ORIGIN_LOOKBACK,PREDICT_STEPS)
    assert_horizon_window_boundaries(test,ends,LOOKBACK,PREDICT_STEPS)
    current=test[TARGET_COL].to_numpy(float); future=test[label].to_numpy(float); delta=future-current
    device=choose_device(); print(f"DEVICE={device} | COMMON_TEST_ORIGINS={len(ends)}")
    rows=[evaluate_one(args,spec,seed,test,ends,current,future,delta,device,seed in save_seeds)
          for spec,seed in all_tasks()]
    runs=pd.DataFrame(rows).sort_values(["feature_set","seed"]).reset_index(drop=True)
    summary=summarize(runs)
    vs_raw=paired_table(runs,[(f"{s['feature_set']}_vs_raw20",str(s["feature_set"]),"raw20")
      for s in FEATURE_SPECS if s["feature_set"]!="raw20"],"test_full_rmse_k")
    component=paired_table(runs,[(f"remove_{g}",f"full20_no_{g}","full20") for g in FEATURE_GROUPS]
      +[("add_optional_apparent_heat_leak","full22_with_apparent_heat_leak","full20")],"test_full_rmse_k")
    out=args.results_dir/"full_test_summary"; out.mkdir(parents=True,exist_ok=True)
    runs.to_csv(out/"test_seed_runs.csv",index=False)
    summary.to_csv(out/"test_feature_summary.csv",index=False)
    vs_raw.to_csv(out/"paired_test_vs_raw20.csv",index=False)
    component.to_csv(out/"paired_test_component_effects.csv",index=False)
    audit={**protocol_payload(),"evaluation_type":"frozen-checkpoint inference",
      "training_or_weight_updates_during_test":False,"test_parent":"0715-BACK",
      "test_variant":"Original","test_label_audit":label_audit,
      "common_test_origins":len(ends),"observed_runs":len(runs),"complete":len(runs)==100,
      "primary_test_metric":"complete-test RMSE","saved_prediction_seeds":sorted(save_seeds),
      "methodological_note":"0715-BACK is a different cooldown-rewarm operating run, but it has been inspected in prior development and is therefore post-hoc rather than pristine."}
    (out/"test_evaluation_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    print(summary.to_string(index=False)); print("FULL_TEST_SUMMARY_COMPLETE=true")


if __name__=="__main__": main()
