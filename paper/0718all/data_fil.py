import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "processed_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

F = 1
PREDICT_STEPS = 60
LOOK_BACK = 60

TRAIN_DATA_FOLDER = ROOT / "300-4.5k-alldataout"
TEST_DATA_FOLDER = ROOT / "300-4.5k original data" 
TRAIN_FILES = [f for f in os.listdir(TRAIN_DATA_FOLDER) if f.lower().endswith(".csv")]
TEST_FILES = ["HTC8300趋势图-F-1226.csv"]

# 独立冷却批次划分：扩增副本与其原始曲线具有相同 source_group，必须进入同一集合。
# 当前 0501 用作验证批次，1226 为最终独立测试批次；0118、0403 用于参数学习。
VAL_SOURCE_GROUPS = {"HTC8300趋势图-F-0501.csv"}
TEST_SOURCE_GROUPS = {"HTC8300趋势图-F-1226.csv"}

# 验证集、测试集保留阀门变化更密集的连续片段。分块而非逐点抽样，避免破坏时序窗口。
# 300 个采样点约为 50 min；每个选中块额外保留历史/预测上下文，且标签不会跨片段。
EVAL_EVENT_BLOCK_STEPS = 300
EVAL_EVENT_FRACTION = 0.35
MIN_EVAL_EVENT_BLOCKS = 3

TARGET_COL = "TE8353"
VALVE_MAIN_COLS = ["CV8300", "CV8313", "CV8310", "CV8351"]
#VALVE_OTHER_COLS = ["CV8312", "CV8311", "CV8359", "CV8350", "CV8330", "CV8339"]
VALVE_OTHER_COLS = ["CV8312", "CV8311"]   #去除信息量较少的阀门列
VALVE_ALL_COLS = VALVE_MAIN_COLS + VALVE_OTHER_COLS
VALVE_THRESHOLD = 0.05


def smooth_sensor_data(data, name):
    """Causal missing-value handling: only carry the most recent past value forward."""
    series = pd.Series(np.asarray(data, dtype=np.float64), copy=True)
    if series.notna().sum() == 0:
        print(f"{name}: all values are missing")
        return series.to_numpy()
    return series.ffill().to_numpy()


def flypoint_filter(data, window_size=5, sg_window=15, sg_poly=2):
    """Trailing median filter. Unlike median/S-G centred filters, it never reads future samples."""
    series = pd.Series(np.asarray(data, dtype=np.float64), copy=True)
    if len(series) < 2:
        return series.to_numpy()
    window_size = max(1, min(window_size, len(series)))
    return series.rolling(window=window_size, min_periods=1).median().to_numpy()


def read_industrial_csv(filepath):
    for enc, sep in [("utf-8", ","), ("utf-8-sig", ","), ("utf-16", "\t"), ("gbk", "\t"), ("gb18030", "\t")]:
        try:
            df = pd.read_csv(filepath, encoding=enc, sep=sep)
            if len(df.columns) > 1:
                df.columns = df.columns.astype(str).str.replace('"', '').str.strip()
                return df
        except Exception:
            pass
    raise ValueError(f"无法正确读取 {filepath}")


def find_col(columns, key):
    matches = [c for c in columns if key.upper() in c.upper() and "TIME" not in c.upper()]
    return matches[0] if matches else None


def get_source_group(fname):
    return re.sub(r"^(Aug_\d+_|Original_)", "", fname, flags=re.IGNORECASE)

#从原始数据中取出并且过滤存到df new中 
def add_numeric_col(df_new, df_raw, raw_col, out_col, window_size=5, sg_window=5):
    series = pd.to_numeric(df_raw[raw_col], errors="coerce").replace(0, np.nan)
    smoothed = smooth_sensor_data(series, out_col)
    df_new[out_col] = flypoint_filter(smoothed, window_size=window_size, sg_window=sg_window, sg_poly=2)


def add_valve_col(df_new, df_raw, raw_col, out_col):
    """阀门开度直接保留原始值：0% 是合法全关状态，不滤波、不平滑、不填补。"""
    df_new[out_col] = pd.to_numeric(df_raw[raw_col], errors="coerce")


def coolprop_pair(temp_values, press_values):
    densities, enthalpies = [], []
    for t, p in zip(temp_values, press_values):
        try:
            if p <= 0 or t <= 0 or pd.isna(p) or pd.isna(t):
                densities.append(np.nan)
                enthalpies.append(np.nan)
            else:
                densities.append(CP.PropsSI("D", "T", t, "P", p * 1e5, "Helium"))
                enthalpies.append(CP.PropsSI("H", "T", t, "P", p * 1e5, "Helium"))
        except Exception:
            densities.append(np.nan)
            enthalpies.append(np.nan)
    return densities, enthalpies


def clean_and_extract(file_list, folder_path, dataset_type):
    all_dfs = []
    for fid, fname in enumerate(file_list):
        filepath = folder_path / fname
        if not filepath.exists():
            print(f"找不到文件: {filepath}")
            continue
        print(f"正在处理 {dataset_type}: {fname}")
        try:
            df_raw = read_industrial_csv(filepath)
        except Exception as e:
            print(e)
            continue

        df_new = pd.DataFrame()
        target_raw_col = find_col(df_raw.columns, TARGET_COL)
        if not target_raw_col:
            print(f"警告: 找不到目标列 {TARGET_COL}")
            continue
        add_numeric_col(df_new, df_raw, target_raw_col, TARGET_COL, window_size=9, sg_window=15)

        valve_actual_cols = []
        for cv in VALVE_ALL_COLS:
            col = find_col(df_raw.columns, cv)
            if col:
                valve_actual_cols.append(col)

        for col in df_raw.columns:
            col_upper = col.upper()
            if "TIME" in col_upper or col == target_raw_col or col in valve_actual_cols:
                continue
            is_temp = "TE" in col_upper or "A" in col_upper or "B" in col_upper
            is_press = "PT" in col_upper or "VAC" in col_upper
            is_flow = "FT" in col_upper
            is_other = "DTBR" in col_upper or "TEF" in col_upper or "TCD" in col_upper
            if is_temp:
                add_numeric_col(df_new, df_raw, col, col, window_size=9, sg_window=15)
            elif is_press or is_flow or is_other:
                add_numeric_col(df_new, df_raw, col, col, window_size=5, sg_window=5)

        event_mask = np.zeros(len(df_raw), dtype=bool)
        cv_cols = {}
        for cv in VALVE_ALL_COLS:
            col = find_col(df_raw.columns, cv)
            if col:
                cv_cols[cv] = col
                add_valve_col(df_new, df_raw, col, col)

        for cv in VALVE_MAIN_COLS:
            col = cv_cols.get(cv)
            if col:
                diff_abs = df_new[col].diff().abs().fillna(0)
                event_mask = np.logical_or(event_mask, diff_abs > VALVE_THRESHOLD)
        df_new["valve_event_mask"] = event_mask.astype(float)     #########################   valve_event_mask   new特征

        if "CV8310" in cv_cols:
            cv8310_col = cv_cols["CV8310"]
            df_new["CV8310_diff"] = df_new[cv8310_col].diff().fillna(0).clip(-5, 5)
            action_points = df_new["CV8310_diff"].abs().values > VALVE_THRESHOLD
            neighborhood = np.zeros(len(df_new), dtype=float)
            for idx in np.where(action_points)[0]:
                # Causal event flag: an action can affect the current and later rows only.
                start = idx
                end = min(len(df_new), idx + LOOK_BACK + 1)
                neighborhood[start:end] = 1.0
            df_new["CV8310_action_neighborhood"] = neighborhood
        else:
            df_new["CV8310_diff"] = 0.0
            df_new["CV8310_action_neighborhood"] = 0.0

        df_new["TE8353_slope_60"] = ((df_new[TARGET_COL] - df_new[TARGET_COL].shift(60)) / 60.0).fillna(0.0)

        required = ["TE8310", "PT8310", "TE8351", "PT8351", "FT8351", "CV8300", "CV8313", "CV8310", "CV8311", "CV8312"]
        found = {key: find_col(df_new.columns, key) for key in required}
        if all(found.values()):
            d8310, h8310 = coolprop_pair(df_new[found["TE8310"]].values, df_new[found["PT8310"]].values)
            d8351, h8351 = coolprop_pair(df_new[found["TE8351"]].values, df_new[found["PT8351"]].values)
            df_new["Fluid_Density_8310"] = pd.Series(d8310).ffill().values
            df_new["Fluid_Enthalpy_8310"] = pd.Series(h8310).ffill().values
            df_new["Fluid_Density_8351"] = pd.Series(d8351).ffill().values
            df_new["Fluid_Enthalpy_8351"] = pd.Series(h8351).ffill().values
            total_cv_open = df_new[found["CV8300"]] + df_new[found["CV8313"]]
            total_cv8310_val = df_new[found["CV8310"]] + df_new[found["CV8311"]] + df_new[found["CV8312"]]
            actual_cv8310_val = (df_new[found["CV8310"]] / total_cv8310_val.replace(0, np.nan)).fillna(0)
            df_new["CV8310_ratio"] = actual_cv8310_val.clip(0, 1)
            df_new["Pseudo_Mass_Flow"] = total_cv_open * np.sqrt(df_new["Fluid_Density_8310"].clip(lower=0))
            df_new["Pseudo_Cooling_Power"] = df_new["Pseudo_Mass_Flow"] * df_new["Fluid_Enthalpy_8310"]
            df_new["Pseudo_Mass_Flow_8310"] = df_new["CV8310_ratio"] * df_new["Pseudo_Mass_Flow"]
            df_new["Pseudo_Cooling_Power_8310"] = df_new["CV8310_ratio"] * df_new["Pseudo_Cooling_Power"]
            df_new["TE8353_TE8310_diff"] = df_new[TARGET_COL] - df_new[found["TE8310"]]
            df_new["Real_Mass_Flow_8351"] = df_new[found["FT8351"]] * df_new["Fluid_Density_8351"]
            df_new["Actual_Cooling_Power_8351"] = df_new["Real_Mass_Flow_8351"] * df_new["Fluid_Enthalpy_8351"]
        else:
            df_new["CV8310_ratio"] = 0.0
            df_new["Pseudo_Mass_Flow_8310"] = 0.0
            df_new["Pseudo_Cooling_Power_8310"] = 0.0
            df_new["TE8353_TE8310_diff"] = 0.0

        label_col = f"Future_Target_{PREDICT_STEPS}step"
        df_new[label_col] = df_new[TARGET_COL].shift(-PREDICT_STEPS)
        df_new["file_id"] = fid
        df_new["source_group"] = get_source_group(fname)
        df_new = df_new.replace([np.inf, -np.inf], np.nan)

        # Causal-only cleaning: never backward-fill a feature or the future label.
        # Raw valve values are not filled; a row with a missing valve/sensor is discarded.
        valve_output_cols = list(cv_cols.values())
        feature_cols = [
            c for c in df_new.columns
            if c not in {label_col, "file_id", "source_group"} and c not in valve_output_cols
        ]
        df_new[feature_cols] = df_new[feature_cols].ffill()
        required_cols = [label_col, *valve_output_cols, *feature_cols]
        df_new = df_new.dropna(subset=required_cols).reset_index(drop=True)
        all_dfs.append(df_new)
        print(f"完成 {fname}: {len(df_new)} 行")

    if not all_dfs:
        return pd.DataFrame()
    final_df = pd.concat(all_dfs, ignore_index=True)
    print(f"{dataset_type} 完成: {len(final_df)} 行, 阀门事件 {final_df['valve_event_mask'].sum():.0f}")
    return final_df


def select_event_dense_segments(df, split_name):
    """按连续块的主阀门事件数排序，构造富含动作的验证/测试集。"""
    if df.empty:
        return df.copy()

    segments = []
    context = LOOK_BACK + PREDICT_STEPS
    for fid, part in df.groupby("file_id", sort=False):
        part = part.reset_index(drop=True)
        n_rows = len(part)
        block_starts = list(range(0, n_rows, EVAL_EVENT_BLOCK_STEPS))
        candidates = []
        for start in block_starts:
            end = min(start + EVAL_EVENT_BLOCK_STEPS, n_rows)
            event_count = float(part.loc[start:end - 1, "valve_event_mask"].sum())
            candidates.append((event_count, start, end))

        n_select = min(
            len(candidates),
            max(MIN_EVAL_EVENT_BLOCKS, int(np.ceil(len(candidates) * EVAL_EVENT_FRACTION))),
        )
        selected = sorted(sorted(candidates, key=lambda item: (-item[0], item[1]))[:n_select], key=lambda item: item[1])

        # Keep sufficient context around each high-event block. Separate segment ids
        # prevent later sequence construction from crossing an omitted interval.
        intervals = []
        for _, start, end in selected:
            left = max(0, start - context)
            right = min(n_rows, end + context)
            if intervals and left <= intervals[-1][1]:
                intervals[-1] = (intervals[-1][0], max(intervals[-1][1], right))
            else:
                intervals.append((left, right))

        for segment_no, (left, right) in enumerate(intervals):
            # Drop the tail so every retained origin has its future label in this segment.
            usable_right = right - PREDICT_STEPS
            if usable_right - left <= LOOK_BACK:
                continue
            segment = part.iloc[left:usable_right].copy()
            segment["file_id"] = f"{fid}__{split_name}_segment_{segment_no}"
            segments.append(segment)

    if not segments:
        raise ValueError(f"{split_name} 未能构造满足历史窗口与预测标签要求的连续片段")

    result = pd.concat(segments, ignore_index=True)
    print(
        f"{split_name} 事件富集完成: {len(result)} 行, "
        f"阀门事件 {result['valve_event_mask'].sum():.0f}, "
        f"事件比例 {result['valve_event_mask'].mean():.4f}"
    )
    return result


def build_splits(df_train_all, df_test_all):
    """Build train, event-enriched validation/test, and a complete held-out test audit set."""
    val_full = df_train_all[df_train_all["source_group"].isin(VAL_SOURCE_GROUPS)].copy()
    train = df_train_all[~df_train_all["source_group"].isin(VAL_SOURCE_GROUPS)].copy()
    test_full = df_test_all[df_test_all["source_group"].isin(TEST_SOURCE_GROUPS)].copy()

    if train.empty or val_full.empty or test_full.empty:
        raise ValueError(
            "训练/验证/测试划分为空，请检查 source_group 名称与 VAL_SOURCE_GROUPS、TEST_SOURCE_GROUPS 配置"
        )

    val = select_event_dense_segments(val_full, "val")
    test = select_event_dense_segments(test_full, "test")
    assert_split_integrity(train, val, test, test_full)
    return train.reset_index(drop=True), val, test, test_full.reset_index(drop=True)


def assert_split_integrity(train, val, test, test_full):
    """Fail fast if source groups or labels violate the split contract."""
    train_groups = set(train["source_group"].unique())
    val_groups = set(val["source_group"].unique())
    test_groups = set(test["source_group"].unique())
    full_test_groups = set(test_full["source_group"].unique())

    if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
        raise AssertionError("source_group overlap detected across train/validation/test")
    if val_groups != VAL_SOURCE_GROUPS:
        raise AssertionError(f"unexpected validation groups: {val_groups}")
    if test_groups != TEST_SOURCE_GROUPS or full_test_groups != TEST_SOURCE_GROUPS:
        raise AssertionError(f"unexpected test groups: event={test_groups}, full={full_test_groups}")

    label_col = f"Future_Target_{PREDICT_STEPS}step"
    for split_name, split_df in {
        "train": train,
        "validation": val,
        "test": test,
        "full_test": test_full,
    }.items():
        if split_df[label_col].isna().any():
            raise AssertionError(f"{split_name} contains missing future labels")
    print(
        "Split integrity passed: "
        f"train={sorted(train_groups)}, val={sorted(val_groups)}, test={sorted(test_groups)}"
    )


if __name__ == "__main__":
    df_train_clean = clean_and_extract(TRAIN_FILES, TRAIN_DATA_FOLDER, "Train")
    df_test_clean = clean_and_extract(TEST_FILES, TEST_DATA_FOLDER, "Test")
    train_split, val_split, test_split, test_full = build_splits(df_train_clean, df_test_clean)

    joblib.dump(train_split, OUT_DIR / "train_clean.pkl")
    joblib.dump(val_split, OUT_DIR / "val_clean.pkl")
    joblib.dump(test_split, OUT_DIR / "test_clean.pkl")
    joblib.dump(test_full, OUT_DIR / "test_full_clean.pkl")
    print(f"训练/验证/测试数据已保存至 {OUT_DIR}")
