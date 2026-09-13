import os
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "processed_data"
SAVE_DIR = ROOT / "saved_models_torch"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

F = 1
PREDICT_STEPS = 90
TARGET_COL = "TE8353" if F else "Thv"
LABEL_COL = f"Future_Target_{PREDICT_STEPS}step"
DELTA_LABEL = "Target_Delta"
LOOK_BACK = 60
BATCH_SIZE = 128
EPOCHS = 200
LEARNING_RATE = 1e-3
PATIENCE = 20
SEED = 42
# 单调性物理约束: 降温阀门开度整窗 +DELTA_RAW(%) 后, 预测的 ΔT 不允许更高
DELTA_RAW = 5.0
LAMBDA_MONO = 1.0
LAMBDA_LB = 0.1
MONO_VALVES = ["CV8310", "CV8300", "CV8351"]


def set_global_seed(seed_value):
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_global_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

print("正在加载已预处理的数据...")
df_train = joblib.load(DATA_DIR / "train_clean.pkl")
df_test = joblib.load(DATA_DIR / "test_clean.pkl")

df_train[DELTA_LABEL] = df_train[LABEL_COL] - df_train[TARGET_COL]
df_test[DELTA_LABEL] = df_test[LABEL_COL] - df_test[TARGET_COL]

ignore_cols = {
    LABEL_COL,
    DELTA_LABEL,
    "valve_event_mask",
    "file_id",
    "source_group",
    "Target_Inertia_10min",
    "Cumulative_Cooling_8351",
    "CV8310_action_neighborhood",
}

ignore_keywords = ["Fluid_Density", "Fluid_Enthalpy","8308", "8309","8390","8391"]

feature_cols = [
    c for c in df_train.columns
    if c not in ignore_cols and not any(k.upper() in c.upper() for k in ignore_keywords)
]

for col in feature_cols:
    if col not in df_test.columns:
        df_test[col] = 0.0

print(f"特征数量: {len(feature_cols)}")
print("所有特征:", feature_cols)


def find_feature_col(key):
    key_up = key.upper()
    for c in feature_cols:
        if c.upper() == f"{key_up} VALUEY":
            return c
    for c in feature_cols:
        cu = c.upper()
        if cu.startswith(f"{key_up} ") and "TIME" not in cu:
            return c
    return None

TE8351_COL = find_feature_col("TE8351")

if TE8351_COL is None:
    raise ValueError("未找到 TE8351 列，无法构建物理下界损失")

print(f"物理下界列: {TE8351_COL}, λ_lb={LAMBDA_LB}")

DROP_TE8351_RAW = True
if DROP_TE8351_RAW and TE8351_COL in feature_cols:
    feature_cols = [c for c in feature_cols if c != TE8351_COL]
    derived_kept = [c for c in feature_cols if "8351" in str(c).upper()]
    print("已经删除TE8351本体")


scaler_x = StandardScaler()
scaler_y = StandardScaler()
train_x_s = scaler_x.fit_transform(df_train[feature_cols]).astype(np.float32)     #训练集
train_y_s = scaler_y.fit_transform(df_train[[DELTA_LABEL]]).astype(np.float32)    #训练集标签
test_x_s = scaler_x.transform(df_test[feature_cols]).astype(np.float32)
test_y_s = scaler_y.transform(df_test[[DELTA_LABEL]]).astype(np.float32)
train_x_s = np.nan_to_num(train_x_s, nan=0.0, posinf=0.0, neginf=0.0)
train_y_s = np.nan_to_num(train_y_s, nan=0.0, posinf=0.0, neginf=0.0)
test_x_s = np.nan_to_num(test_x_s, nan=0.0, posinf=0.0, neginf=0.0)
test_y_s = np.nan_to_num(test_y_s, nan=0.0, posinf=0.0, neginf=0.0)

def create_sequences(df_ref, x_data, y_delta_scaled):   #（所有训练集数据、训练集标准化后的、训练集标签标准化后的）
    x_seq, x_mask, x_aux, y_delta, y_curr_abs, y_fut_abs, y_lb_abs, sample_groups = [], [], [], [], [], [], [], []
    curr_temp_raw = df_ref[TARGET_COL].values
    fut_temp_raw = df_ref[LABEL_COL].values
    lb_temp_raw = df_ref[TE8351_COL].values #下界
    has_source_group = "source_group" in df_ref.columns
    if "Fluid_Density_8310" in df_ref.columns and "Fluid_Enthalpy_8310" in df_ref.columns:
        rho = pd.to_numeric(df_ref["Fluid_Density_8310"], errors="coerce").fillna(0.0).values
        enth = pd.to_numeric(df_ref["Fluid_Enthalpy_8310"], errors="coerce").fillna(0.0).values
        a0 = np.sqrt(np.clip(rho, 0.0, None)) #np.sqrt开根号，密度开根号后与流量成正比，作为扰动时的物理缩放因子 np.clip进行数值裁剪，避免出现负值
        aux_full = np.stack([a0, a0 * enth], axis=1).astype(np.float32)
    else:
        aux_full = np.zeros((len(df_ref), 2), dtype=np.float32)

    for fid in df_ref["file_id"].unique():
        idx = df_ref["file_id"] == fid
        x_sub = x_data[idx]
        y_delta_sub = y_delta_scaled[idx]
        mask_sub = df_ref.loc[idx, "valve_event_mask"].values
        curr_sub = curr_temp_raw[idx]
        fut_sub = fut_temp_raw[idx]
        lb_sub = lb_temp_raw[idx]
        aux_sub = aux_full[np.asarray(idx)]
        group_value = df_ref.loc[idx, "source_group"].iloc[0] if has_source_group else str(fid)
        for i in range(len(x_sub) - LOOK_BACK - PREDICT_STEPS + 1):
            end_idx = i + LOOK_BACK - 1
            future_idx = end_idx + PREDICT_STEPS
            if np.isnan(y_delta_sub[end_idx]).any() or np.isnan(fut_sub[end_idx]) or np.isnan(lb_sub[future_idx]):
                continue
            x_seq.append(x_sub[i:i + LOOK_BACK])
            x_mask.append(mask_sub[i:i + LOOK_BACK].reshape(-1, 1))
            x_aux.append(aux_sub[i:i + LOOK_BACK])
            y_delta.append(y_delta_sub[end_idx])
            y_curr_abs.append(curr_sub[end_idx])
            y_fut_abs.append(fut_sub[end_idx])
            y_lb_abs.append(lb_sub[future_idx])
            sample_groups.append(group_value)
    return (
        np.array(x_seq, dtype=np.float32),
        np.array(x_mask, dtype=np.float32),
        np.array(x_aux, dtype=np.float32),
        np.array(y_delta, dtype=np.float32),
        np.array(y_curr_abs, dtype=np.float32),
        np.array(y_fut_abs, dtype=np.float32),
        np.array(y_lb_abs, dtype=np.float32),
        np.array(sample_groups),
    )


print("构建时序样本")
X_train_full, M_train_full, AUX_train_full, y_delta_full, y_curr_full, y_fut_full, y_lb_full, group_full = create_sequences(df_train, train_x_s, train_y_s)
X_test, M_test, _, y_delta_test, y_curr_test, y_fut_test, y_lb_test, _ = create_sequences(df_test, test_x_s, test_y_s)

unique_groups = np.array(sorted(pd.unique(group_full)))
if len(unique_groups) < 2:
    raise ValueError("source_group 数量少于 2，无法按原始文件组划分训练/验证集")
val_group_count = max(1, int(np.ceil(len(unique_groups) * 0.25)))
val_groups = set(unique_groups[-val_group_count:])
train_mask = np.array([g not in val_groups for g in group_full])
val_mask = ~train_mask
print(f"按 source_group 划分: 训练组={sorted(set(unique_groups) - val_groups)}, 验证组={sorted(val_groups)}")

X_t, X_v = X_train_full[train_mask], X_train_full[val_mask]
M_t, M_v = M_train_full[train_mask], M_train_full[val_mask]
AUX_t, AUX_v = AUX_train_full[train_mask], AUX_train_full[val_mask]
y_t, y_v = y_delta_full[train_mask], y_delta_full[val_mask]
y_c_t, y_c_v = y_curr_full[train_mask], y_curr_full[val_mask]
y_f_t, y_f_v = y_fut_full[train_mask], y_fut_full[val_mask]
y_lb_t, y_lb_v = y_lb_full[train_mask], y_lb_full[val_mask]

sample_weights = np.ones(len(y_t), dtype=np.float32)
true_delta_raw = y_f_t - y_c_t
temp_changing = np.where(np.abs(true_delta_raw) > 0.5)[0]
sample_weights[temp_changing] = np.maximum(sample_weights[temp_changing], 5.0)
high_head = int(len(y_t) * 0.25)
high_tail = int(len(y_t) * 0.85)
sample_weights[:high_head] = np.maximum(sample_weights[:high_head], 10.0)
sample_weights[high_tail:] = np.maximum(sample_weights[high_tail:], 10.0)

train_dataset = TensorDataset(
    torch.from_numpy(X_t),
    torch.from_numpy(M_t),
    torch.from_numpy(AUX_t),
    torch.from_numpy(y_t),
    torch.from_numpy(sample_weights.reshape(-1, 1)),
    torch.from_numpy(y_c_t),
    torch.from_numpy(y_lb_t),
)
val_dataset = TensorDataset(
    torch.from_numpy(X_v),
    torch.from_numpy(M_v),
    torch.from_numpy(AUX_v),
    torch.from_numpy(y_v),
    torch.from_numpy(y_c_v),
    torch.from_numpy(y_lb_v),
)
test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(M_test), torch.from_numpy(y_delta_test))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available())
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available())


class PriorGuidedAttention(nn.Module):
    def __init__(self, hidden_size, time_steps):
        super().__init__()
        self.att_weight = nn.Linear(hidden_size, 1, bias=False)
        self.att_bias = nn.Parameter(torch.zeros(time_steps, 1))
        self.event_boost = nn.Parameter(torch.tensor([0.02], dtype=torch.float32))

    def forward(self, gru_out, event_mask):
        e = torch.tanh(self.att_weight(gru_out) + self.att_bias.unsqueeze(0))
        e_boosted = e + event_mask * self.event_boost
        alpha = torch.softmax(e_boosted, dim=1)
        context = torch.sum(gru_out * alpha, dim=1)
        return context, alpha


class GRUTemperatureModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, time_steps=60, dropout=0.2):
        super().__init__()
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.gru2 = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.attention = PriorGuidedAttention(hidden_size, time_steps)
        self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, mask):
        out, _ = self.gru1(x)
        out = self.dropout(out)
        out, _ = self.gru2(out)
        context, alpha = self.attention(out, mask)
        return self.head(context), alpha


model = GRUTemperatureModel(input_size=len(feature_cols), time_steps=LOOK_BACK).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ===== 单调性物理约束: 反事实扰动 =====
# 训练时把降温阀门开度整窗 +DELTA_RAW%, 并按 data_fil.py 的公式同步重算衍生特征链
# (与 run.py 的干预算子一致, 防止模型把反向依赖藏进未被扰动的衍生列里),
# 然后惩罚 ReLU(f(扰动) - f(原始)): 开大降温阀后预测的 ΔT 不允许更高。
# 整窗平移下 CV8310_diff / valve_event_mask / TE8353_slope_60 严格不变, 无需更新。
pert_names = {
    "CV8300": find_feature_col("CV8300"),
    "CV8313": find_feature_col("CV8313"),
    "CV8310": find_feature_col("CV8310"),
    "CV8311": find_feature_col("CV8311"),
    "CV8312": find_feature_col("CV8312"),
    "CV8351": find_feature_col("CV8351"),
}
for _derived in ["CV8310_ratio", "Pseudo_Mass_Flow", "Pseudo_Cooling_Power", "Pseudo_Mass_Flow_8310", "Pseudo_Cooling_Power_8310"]:
    pert_names[_derived] = _derived if _derived in feature_cols else None
PERT_IDX = {k: feature_cols.index(v) for k, v in pert_names.items() if v is not None}
missing = [k for k, v in pert_names.items() if v is None]
if missing:
    print(f"[warn] 扰动相关列缺失, 对应衍生特征不会被重算: {missing}")
print(f"单调性约束阀门: {MONO_VALVES}, δ=+{DELTA_RAW}%, λ={LAMBDA_MONO}")

FEAT_MEAN = torch.tensor(scaler_x.mean_, dtype=torch.float32, device=device)
FEAT_SCALE = torch.tensor(scaler_x.scale_, dtype=torch.float32, device=device)
MEAN_Y = float(scaler_y.mean_[0])
SCALE_Y = float(scaler_y.scale_[0])


def _destd(x_std, i):
    return x_std[..., i] * FEAT_SCALE[i] + FEAT_MEAN[i]


def _std(raw, i):
    return (raw - FEAT_MEAN[i]) / FEAT_SCALE[i]


def perturb_valve(x_std, aux, valve):
    x = x_std.clone()
    i_v = PERT_IDX[valve]
    v_new = (_destd(x_std, i_v) + DELTA_RAW).clamp(0.0, 100.0)
    x[..., i_v] = _std(v_new, i_v)
    if valve == "CV8310" and all(k in PERT_IDX for k in ["CV8311", "CV8312", "CV8310_ratio", "Pseudo_Mass_Flow", "Pseudo_Cooling_Power", "Pseudo_Mass_Flow_8310", "Pseudo_Cooling_Power_8310"]):
        # data_fil.py:183-189 — ratio = CV8310/(CV8310+CV8311+CV8312), 伪冷量 8310 = ratio × G管伪冷量
        denom = v_new + _destd(x_std, PERT_IDX["CV8311"]) + _destd(x_std, PERT_IDX["CV8312"])
        ratio = torch.where(denom.abs() > 1e-6, v_new / denom, torch.zeros_like(denom)).clamp(0.0, 1.0)
        pmf = _destd(x_std, PERT_IDX["Pseudo_Mass_Flow"])
        pcp = _destd(x_std, PERT_IDX["Pseudo_Cooling_Power"])
        x[..., PERT_IDX["CV8310_ratio"]] = _std(ratio, PERT_IDX["CV8310_ratio"])
        x[..., PERT_IDX["Pseudo_Mass_Flow_8310"]] = _std(ratio * pmf, PERT_IDX["Pseudo_Mass_Flow_8310"])
        x[..., PERT_IDX["Pseudo_Cooling_Power_8310"]] = _std(ratio * pcp, PERT_IDX["Pseudo_Cooling_Power_8310"])
    elif valve == "CV8300" and all(k in PERT_IDX for k in ["CV8313", "CV8310_ratio", "Pseudo_Mass_Flow", "Pseudo_Cooling_Power", "Pseudo_Mass_Flow_8310", "Pseudo_Cooling_Power_8310"]):
        # data_fil.py:182,186-189 — G管伪流量 = (CV8300+CV8313)·√ρ, 伪冷量 = 伪流量·H, 连锁更新 8310 两列
        # 用 aux 物性量从头重算而非乘性缩放旧值: 旧值为 0 时 (阀门从关到开) 乘性更新会失效
        total = v_new + _destd(x_std, PERT_IDX["CV8313"])
        pmf_new = total * aux[..., 0]
        pcp_new = total * aux[..., 1]
        ratio = _destd(x_std, PERT_IDX["CV8310_ratio"]).clamp(0.0, 1.0)
        x[..., PERT_IDX["Pseudo_Mass_Flow"]] = _std(pmf_new, PERT_IDX["Pseudo_Mass_Flow"])
        x[..., PERT_IDX["Pseudo_Cooling_Power"]] = _std(pcp_new, PERT_IDX["Pseudo_Cooling_Power"])
        x[..., PERT_IDX["Pseudo_Mass_Flow_8310"]] = _std(ratio * pmf_new, PERT_IDX["Pseudo_Mass_Flow_8310"])
        x[..., PERT_IDX["Pseudo_Cooling_Power_8310"]] = _std(ratio * pcp_new, PERT_IDX["Pseudo_Cooling_Power_8310"])
    # CV8351: 无衍生特征, 仅扰动裸开度列 (纳入约束以保护其现有正确行为)
    return x
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)
criterion = nn.HuberLoss(delta=1.0, reduction="none")

best_val = float("inf")
best_state = None
best_viol_rates = {}
wait = 0
history_train, history_val = [], []

print("开始训练 PyTorch GRU ΔT 模型 (含单调性物理约束 + TE8351 下界约束)...")
n_blocks = 1 + len(MONO_VALVES)
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses, train_monos, train_lbs = [], [], []
    for xb, mb, auxb, yb, wb, ycb, tlbb in train_loader:
        xb = xb.to(device, non_blocking=True)
        mb = mb.to(device, non_blocking=True)
        auxb = auxb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        wb = wb.to(device, non_blocking=True)
        ycb = ycb.to(device, non_blocking=True)
        tlbb = tlbb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        # [原始; 扰CV8310; 扰CV8300; 扰CV8351] 拼成大 batch 单次前向, 监督损失只用原始块
        x_all = torch.cat([xb] + [perturb_valve(xb, auxb, v) for v in MONO_VALVES], dim=0)
        m_all = mb.repeat(n_blocks, 1, 1)
        pred_all, _ = model(x_all, m_all)
        n = xb.size(0)
        pred_base = pred_all[:n]
        sup_loss = (criterion(pred_base, yb) * wb).mean()
        mono_loss = pred_base.new_zeros(())
        for k in range(len(MONO_VALVES)):
            mono_loss = mono_loss + torch.relu(pred_all[n * (k + 1):n * (k + 2)] - pred_base).mean()
        pred_abs = ycb + pred_base.squeeze(-1) * SCALE_Y + MEAN_Y
        lb_loss = torch.relu(tlbb - pred_abs).mean()
        loss = sup_loss + LAMBDA_MONO * mono_loss + LAMBDA_LB * lb_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(sup_loss.item())
        train_monos.append(mono_loss.item())
        train_lbs.append(lb_loss.item())

    model.eval()
    val_losses, val_monos, val_lbs = [], [], []
    viol_sum = {v: 0.0 for v in MONO_VALVES}
    viol_cnt = {v: 0 for v in MONO_VALVES}
    n_val = 0
    with torch.no_grad():
        for xb, mb, auxb, yb, ycb, tlbb in val_loader:
            xb = xb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            auxb = auxb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            ycb = ycb.to(device, non_blocking=True)
            tlbb = tlbb.to(device, non_blocking=True)
            x_all = torch.cat([xb] + [perturb_valve(xb, auxb, v) for v in MONO_VALVES], dim=0)
            m_all = mb.repeat(n_blocks, 1, 1)
            pred_all, _ = model(x_all, m_all)
            n = xb.size(0)
            pred_base = pred_all[:n]
            pred_abs = ycb + pred_base.squeeze(-1) * SCALE_Y + MEAN_Y
            val_lbs.append(torch.relu(tlbb - pred_abs).mean().item())
            val_losses.append(criterion(pred_base, yb).mean().item())
            batch_mono = 0.0
            for k, v in enumerate(MONO_VALVES):
                diff = pred_all[n * (k + 1):n * (k + 2)] - pred_base
                viol_cnt[v] += int((diff > 0).sum().item())
                viol_sum[v] += float(torch.relu(diff).sum().item())
                batch_mono += float(torch.relu(diff).mean().item())
            val_monos.append(batch_mono)
            n_val += n

    train_loss = float(np.mean(train_losses))
    train_mono = float(np.mean(train_monos))
    train_lb = float(np.mean(train_lbs))
    val_loss = float(np.mean(val_losses))
    val_mono = float(np.mean(val_monos))
    val_lb = float(np.mean(val_lbs))
    val_metric = val_loss + LAMBDA_MONO * val_mono + LAMBDA_LB * val_lb
    history_train.append(train_loss)
    history_val.append(val_metric)
    scheduler.step(val_metric)
    scale_y_k = float(scaler_y.scale_[0])
    viol_str = " ".join(
        f"{v}:{viol_cnt[v] / max(n_val, 1) * 100:.1f}%/{viol_sum[v] / max(viol_cnt[v], 1) * scale_y_k:.3f}K"
        for v in MONO_VALVES
    )
    print(
        f"Epoch {epoch:03d}/{EPOCHS} - train: {train_loss:.6f} (mono {train_mono:.6f}, lb {train_lb:.3f}K) "
        f"- val: {val_loss:.6f} (mono {val_mono:.6f}, lb {val_lb:.3f}K) - 违反率/均幅 {viol_str}"
    )

    if val_metric < best_val:
        best_val = val_metric
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_viol_rates = {v: viol_cnt[v] / max(n_val, 1) for v in MONO_VALVES}
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print("EarlyStopping 触发")
            break

if best_state is not None:
    model.load_state_dict(best_state)

model.eval()
preds = []
with torch.no_grad():
    for xb, mb, _ in test_loader:
        xb = xb.to(device, non_blocking=True)
        mb = mb.to(device, non_blocking=True)
        pred, _ = model(xb, mb)
        preds.append(pred.cpu().numpy())
y_pred_delta_s = np.vstack(preds)
y_pred_delta_raw = scaler_y.inverse_transform(y_pred_delta_s).flatten()
y_pred_abs = y_curr_test + y_pred_delta_raw
y_real_abs = y_fut_test
mae = mean_absolute_error(y_real_abs, y_pred_abs)
r2 = r2_score(y_real_abs, y_pred_abs)
print(f"测试集最终评估 -> MAE: {mae:.2f} K, R²: {r2:.4f}")

plt.figure(figsize=(14, 8))
ax1 = plt.subplot(2, 1, 1)
ax1.plot(y_real_abs, label="Actual Future Temperature", color="blue", linewidth=1.5)
ax1.plot(y_pred_abs, label="Predicted Future Temperature", color="red", linestyle="--", linewidth=1.5)
ax1.set_title(f"PyTorch Delta-GRU Accuracy ({PREDICT_STEPS} Steps Ahead)\nMAE: {mae:.2f} K | R²: {r2:.4f}")
ax1.set_ylabel("Temperature [K]")
ax1.legend(loc="upper right")
ax1.grid(True, linestyle="--", alpha=0.6)
ax2 = plt.subplot(2, 1, 2, sharex=ax1)
event_flags = M_test[:, -1, 0]
event_indices = np.where(event_flags > 0)[0]
ax2.vlines(event_indices, ymin=0, ymax=1, color="orange", alpha=0.5, label="Valve Action Trigger")
ax2.set_yticks([])
ax2.set_xlabel("Time Steps")
ax2.set_ylabel("Valve Events")
ax2.legend(loc="upper right")
plt.tight_layout()
plot_path = ROOT / f"torch_delta_prediction_accuracy_ahead{PREDICT_STEPS}.png"
plt.savefig(plot_path, dpi=200)
print(f"预测图已保存: {plot_path}")

model_path = SAVE_DIR / "gru_temperature_model_torch.pt"
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "look_back": LOOK_BACK,
        "predict_steps": PREDICT_STEPS,
        "target_col": TARGET_COL,
        "mae": mae,
        "r2": r2,
        "lambda_mono": LAMBDA_MONO,
        "lambda_lb": LAMBDA_LB,
        "te8351_lower_bound_col": TE8351_COL,
        "delta_raw": DELTA_RAW,
        "mono_valves": MONO_VALVES,
        "val_violation_rates": best_viol_rates,
    },
    model_path,
)
joblib.dump(scaler_x, SAVE_DIR / "scaler_x.pkl")
joblib.dump(scaler_y, SAVE_DIR / "scaler_y.pkl")
joblib.dump(feature_cols, SAVE_DIR / "feature_cols.pkl")
print(f"模型与 Scaler 已保存至: {SAVE_DIR}")
