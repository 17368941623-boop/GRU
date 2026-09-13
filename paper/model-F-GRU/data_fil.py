import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import medfilt, savgol_filter
import CoolProp.CoolProp as CP

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "processed_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

F = 1
PREDICT_STEPS = 90
LOOK_BACK = 60

TRAIN_DATA_FOLDER = ROOT / "300-4.5k-alldataout"
TEST_DATA_FOLDER = ROOT / "300-4.5k original data" 
TRAIN_FILES = [f for f in os.listdir(TRAIN_DATA_FOLDER) if f.lower().endswith(".csv")]
TEST_FILES = ["HTC8300趋势图-F-1226.csv"]

TARGET_COL = "TE8353"
VALVE_MAIN_COLS = ["CV8300", "CV8313", "CV8310", "CV8351"]
#VALVE_OTHER_COLS = ["CV8312", "CV8311", "CV8359", "CV8350", "CV8330", "CV8339"]
VALVE_OTHER_COLS = ["CV8312", "CV8311"]   #去除信息量较少的阀门列
VALVE_ALL_COLS = VALVE_MAIN_COLS + VALVE_OTHER_COLS
VALVE_THRESHOLD = 0.05


def smooth_sensor_data(data, name):
    arr = np.array(data, dtype=np.float64, copy=True)
    if len(arr) == 0:
        return arr
    if pd.isna(arr[0]):
        valid_indices = np.where(~np.isnan(arr))[0]
        if len(valid_indices) > 0:
            arr[0] = arr[valid_indices[0]]
        else:
            print(f"{name}: 全是nan值无数据可用")
            return arr
    for i in range(1, len(arr)):
        if pd.isna(arr[i]):
            arr[i] = arr[i - 1]
    return arr


def flypoint_filter(data, window_size=5, sg_window=15, sg_poly=2):
    arr = np.array(data, dtype=np.float64, copy=True)
    if len(arr) < 5:
        return arr
    window_size = min(window_size, len(arr) if len(arr) % 2 == 1 else len(arr) - 1)
    sg_window = min(sg_window, len(arr) if len(arr) % 2 == 1 else len(arr) - 1)
    if window_size < 3 or sg_window < 3:
        return arr
    filtered_arr = medfilt(arr, kernel_size=window_size)
    return savgol_filter(filtered_arr, window_length=sg_window, polyorder=min(sg_poly, sg_window - 1))


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
                add_numeric_col(df_new, df_raw, col, col, window_size=5, sg_window=5)

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
                start = max(0, idx - LOOK_BACK)
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
            df_new["Fluid_Density_8310"] = pd.Series(d8310).ffill().bfill().values
            df_new["Fluid_Enthalpy_8310"] = pd.Series(h8310).ffill().bfill().values
            df_new["Fluid_Density_8351"] = pd.Series(d8351).ffill().bfill().values
            df_new["Fluid_Enthalpy_8351"] = pd.Series(h8351).ffill().bfill().values
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
        df_new = df_new.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna(subset=[label_col]).reset_index(drop=True)
        all_dfs.append(df_new)
        print(f"完成 {fname}: {len(df_new)} 行")

    if not all_dfs:
        return pd.DataFrame()
    final_df = pd.concat(all_dfs, ignore_index=True)
    print(f"{dataset_type} 完成: {len(final_df)} 行, 阀门事件 {final_df['valve_event_mask'].sum():.0f}")
    return final_df


if __name__ == "__main__":
    df_train_clean = clean_and_extract(TRAIN_FILES, TRAIN_DATA_FOLDER, "Train")
    df_test_clean = clean_and_extract(TEST_FILES, TEST_DATA_FOLDER, "Test")
    joblib.dump(df_train_clean, OUT_DIR / "train_clean.pkl")
    joblib.dump(df_test_clean, OUT_DIR / "test_clean.pkl")
    print(f"所有数据已保存至 {OUT_DIR}")
