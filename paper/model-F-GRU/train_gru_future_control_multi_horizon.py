"""
Future-control-conditioned multi-horizon GRU diagnostic.

This model receives the historical sensor window and the recorded future
valve sequence. It is an oracle/conditional-control experiment: for online
deployment, replace the recorded future controls with an MPC candidate
sequence or a separate valve-policy prediction.

Run data preprocessing again after fixing data_fil.py so valid valve 0% values
are not replaced with NaN and forward-filled.
"""
import os
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "processed_data"
SAVE_DIR = ROOT / "saved_models_torch_future_control_multi"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "TE8353"
LOOK_BACK = 60
HORIZONS = [10, 20, 30, 45, 60, 75, 90]
MAX_HORIZON = max(HORIZONS)
BATCH_SIZE = 128
EPOCHS = 200
LEARNING_RATE = 1e-3
PATIENCE = 20
HIDDEN_SIZE = 128
SEED = 42
HORIZON_WEIGHTS = np.asarray([0.25, 0.40, 0.60, 0.85, 1.00, 1.20, 1.50], dtype=np.float32)
LAMBDA_RATE_SMOOTH = 0.01
# 当前实验目标是验证“给定真实未来控制轨迹时”的温度预测能力，
# 因此训练阶段默认始终使用原始数据中的真实未来控制量。
# 若之后要做“无未来控制量”的鲁棒性实验，可单独改为 0.50 或 1.0。
FUTURE_CONTROL_DROP_PROB = 0.0
FUTURE_CONTROL_FALLBACK = "hold_last"
PRIMARY_VALIDATION_MODE = "recorded"

CONTROL_KEYS = ["CV8300", "CV8313", "CV8310", "CV8311", "CV8312", "CV8351"]
CONTROL_EVENT_THRESHOLD = 0.05


def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def find_feature_col(columns, key):
    key_upper = key.upper()
    exact = [c for c in columns if str(c).upper() == f"{key_upper} VALUEY"]
    if exact:
        return exact[0]
    starts = [
        c for c in columns
        if str(c).upper().startswith(f"{key_upper} ")
        and "TIME" not in str(c).upper()
    ]
    return starts[0] if starts else None


def build_features(df_train, df_test):
    future_cols = {c for c in df_train.columns if str(c).startswith("Future_Target_")}
    ignore = {
        *future_cols,
        "Target_Delta",
        "valve_event_mask",
        "file_id",
        "source_group",
        "Target_Inertia_10min",
        "Cumulative_Cooling_8351",
        "CV8310_action_neighborhood",
        "TE8353_slope_60",
    }
    feature_cols = [
        c for c in df_train.columns
        if c not in ignore
        and "FLUID_DENSITY" not in str(c).upper()
        and "FLUID_ENTHALPY" not in str(c).upper()
    ]
    for col in feature_cols:
        if col not in df_test.columns:
            df_test[col] = 0.0

    control_cols = []
    for key in CONTROL_KEYS:
        col = find_feature_col(feature_cols, key)
        if col is None:
            raise ValueError(f"Cannot find control feature: {key}")
        control_cols.append(col)
    control_indices = [feature_cols.index(c) for c in control_cols]
    return feature_cols, control_cols, control_indices


def split_groups(df):
    groups = np.asarray(sorted(pd.unique(df["source_group"])))
    if len(groups) < 2:
        raise ValueError("At least two source_group values are required")
    n_val = max(1, int(np.ceil(len(groups) * 0.25)))
    val_groups = set(groups[-n_val:])
    train_groups = set(groups) - val_groups
    print("Train groups:", sorted(train_groups))
    print("Validation groups:", sorted(val_groups))
    return train_groups, val_groups


def create_sequences(df, x_scaled, feature_cols, control_cols, control_indices):
    x_seq, m_seq, u_recorded_seq = [], [], []
    y_delta, y_current, y_future_main = [], [], []
    groups, file_ids, origins, dynamic = [], [], [], []

    target = pd.to_numeric(df[TARGET_COL], errors="coerce").to_numpy(float)
    # 按 file_id 填补控制量，避免在不同原始文件之间发生前后文件串值。
    controls_raw_frame = df[control_cols].apply(pd.to_numeric, errors="coerce")
    controls_raw_full = np.zeros((len(df), len(control_cols)), dtype=float)
    unique_fids = df["file_id"].unique()
    for fid in unique_fids:
        file_mask = df["file_id"].to_numpy() == fid
        controls_raw_full[file_mask] = (
            controls_raw_frame.loc[file_mask]
            .ffill()
            .bfill()
            .fillna(0.0)
            .to_numpy(float)
        )
    controls_scaled_full = x_scaled[:, control_indices]
    file_id_values = df["file_id"].to_numpy()

    for file_no, fid in enumerate(unique_fids, start=1):
        row_mask = file_id_values == fid
        row_indices = np.flatnonzero(row_mask)
        x_sub = x_scaled[row_mask]
        target_sub = target[row_mask]
        controls_raw = controls_raw_full[row_mask]
        controls_scaled = controls_scaled_full[row_mask]
        event_mask = df.loc[row_mask, "valve_event_mask"].to_numpy(float)
        group_value = df.loc[row_mask, "source_group"].iloc[0]

        n_samples = len(x_sub) - LOOK_BACK - MAX_HORIZON + 1
        for i in range(max(0, n_samples)):
            end = i + LOOK_BACK - 1
            future_end = end + MAX_HORIZON
            if np.isnan(target_sub[end]) or np.isnan(target_sub[future_end]):
                continue

            current = target_sub[end]
            future_values = np.asarray(
                [target_sub[end + h] for h in HORIZONS], dtype=np.float32
            )
            delta = future_values - current

            # U[t], U[t+1], ..., U[t+MAX_HORIZON-1] drive transitions
            # t->t+1, ..., t+MAX_HORIZON-1->t+MAX_HORIZON.
            u_future = controls_scaled[end:future_end]
            u_future_raw = controls_raw[end:future_end]
            valve_event = np.any(
                np.abs(np.diff(u_future_raw, axis=0)) > CONTROL_EVENT_THRESHOLD,
                axis=1,
            )
            is_dynamic = bool(np.max(np.abs(delta)) > 0.5 or np.any(valve_event))

            x_seq.append(x_sub[i:i + LOOK_BACK])
            m_seq.append(event_mask[i:i + LOOK_BACK].reshape(-1, 1))
            # 最后一列是 future_control_known 标志：
            # 1 = 真实/计划未来控制量，0 = 当前控制量保持不变的替代序列。
            u_recorded_seq.append(
                np.concatenate(
                    [u_future, np.ones((MAX_HORIZON, 1), dtype=np.float32)],
                    axis=1,
                )
            )
            y_delta.append(delta)
            y_current.append(current)
            y_future_main.append(target_sub[end + HORIZONS[-1]])
            groups.append(group_value)
            file_ids.append(fid)
            origins.append(int(row_indices[end]))
            dynamic.append(is_dynamic)

        if file_no == 1 or file_no == len(unique_fids) or file_no % 5 == 0:
            print(
                f"  sequence files {file_no}/{len(unique_fids)} "
                f"({len(x_seq):,} samples)",
                flush=True,
            )

    return (
        np.asarray(x_seq, dtype=np.float32),
        np.asarray(m_seq, dtype=np.float32),
        np.asarray(u_recorded_seq, dtype=np.float32),
        np.asarray(y_delta, dtype=np.float32),
        np.asarray(y_current, dtype=np.float32),
        np.asarray(y_future_main, dtype=np.float32),
        np.asarray(groups),
        np.asarray(file_ids),
        np.asarray(origins),
        np.asarray(dynamic, dtype=bool),
    )


class PriorGuidedAttention(nn.Module):
    def __init__(self, hidden_size, time_steps):
        super().__init__()
        self.att_weight = nn.Linear(hidden_size, 1, bias=False)
        self.att_bias = nn.Parameter(torch.zeros(time_steps, 1))
        self.event_boost = nn.Parameter(torch.tensor([0.02], dtype=torch.float32))

    def forward(self, gru_out, event_mask):
        score = torch.tanh(
            self.att_weight(gru_out) + self.att_bias.unsqueeze(0)
        )
        score = score + event_mask * self.event_boost
        alpha = torch.softmax(score, dim=1)
        context = torch.sum(gru_out * alpha, dim=1)
        return context, alpha


class FutureControlGRU(nn.Module):
    def __init__(self, input_size, control_size, hidden_size=128, time_steps=60):
        super().__init__()
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.attention = PriorGuidedAttention(hidden_size, time_steps)
        self.decoder = nn.GRUCell(control_size, hidden_size)
        self.rate_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, event_mask, future_controls):
        out, _ = self.gru1(x)
        out = self.dropout(out)
        out, _ = self.gru2(out)
        context, alpha = self.attention(out, event_mask)

        hidden = context
        rates = []
        for step in range(future_controls.shape[1]):
            hidden = self.decoder(future_controls[:, step, :], hidden)
            rates.append(self.rate_head(hidden))

        rates = torch.stack(rates, dim=1)
        cumulative_delta = torch.cumsum(rates, dim=1)
        horizon_idx = torch.tensor(
            [h - 1 for h in HORIZONS],
            dtype=torch.long,
            device=x.device,
        )
        horizon_delta = cumulative_delta.index_select(1, horizon_idx).squeeze(-1)
        return horizon_delta, rates.squeeze(-1), alpha


def compute_loss(pred_delta, rates, target_delta, sample_weight, horizon_weights):
    loss_level = nn.functional.huber_loss(
        pred_delta,
        target_delta,
        reduction="none",
        delta=1.0,
    )
    loss_level = loss_level * horizon_weights.view(1, -1)
    loss_level = (loss_level * sample_weight).mean()
    loss_rate = torch.mean((rates[:, 1:] - rates[:, :-1]) ** 2)
    return loss_level + LAMBDA_RATE_SMOOTH * loss_rate


def build_hold_future_controls(x, control_indices, horizon):
    """在 batch 内构造 hold-last 控制序列，避免预先复制整套数据。"""
    idx = torch.as_tensor(control_indices, dtype=torch.long, device=x.device)
    current = x[:, -1, :].index_select(1, idx)
    u_hold = current.unsqueeze(1).expand(-1, horizon, -1)
    known = torch.zeros(
        (x.shape[0], horizon, 1), dtype=x.dtype, device=x.device
    )
    return torch.cat([u_hold, known], dim=-1)


def choose_future_controls(u_recorded, u_hold, drop_prob, training):
    """选择未来控制量输入；训练时随机丢弃真实未来控制量。"""
    if not training or drop_prob <= 0.0:
        return u_recorded
    use_hold = torch.rand(
        (u_recorded.shape[0], 1, 1), device=u_recorded.device
    ) < drop_prob
    return torch.where(use_hold, u_hold, u_recorded)


set_global_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, flush=True)

print("Loading preprocessed data...", flush=True)
df_train = joblib.load(DATA_DIR / "train_clean.pkl")
df_test = joblib.load(DATA_DIR / "test_clean.pkl")
print(
    f"Loaded train={len(df_train):,} rows, test={len(df_test):,} rows",
    flush=True,
)
feature_cols, control_cols, control_indices = build_features(df_train, df_test)
train_groups, val_groups = split_groups(df_train)

scaler_x = StandardScaler()
fit_rows = df_train["source_group"].isin(train_groups)
scaler_x.fit(df_train.loc[fit_rows, feature_cols])
train_x = np.nan_to_num(
    scaler_x.transform(df_train[feature_cols]).astype(np.float32),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)
test_x = np.nan_to_num(
    scaler_x.transform(df_test[feature_cols]).astype(np.float32),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)

print("Building sequences...", flush=True)
train_all = create_sequences(
    df_train, train_x, feature_cols, control_cols, control_indices
)
test_all = create_sequences(
    df_test, test_x, feature_cols, control_cols, control_indices
)

(
    X_full, M_full, U_recorded_full,
    y_raw_full, y_curr_full, y_fut_full,
    group_full, _, _, dynamic_full,
) = train_all
(
    X_test, M_test, U_recorded_test,
    y_raw_test, y_curr_test, y_fut_test,
    group_test, file_test, origin_test, dynamic_test,
) = test_all

train_mask = np.asarray([g in train_groups for g in group_full])
val_mask = np.asarray([g in val_groups for g in group_full])

X_t, X_v = X_full[train_mask], X_full[val_mask]
M_t, M_v = M_full[train_mask], M_full[val_mask]
U_recorded_t, U_recorded_v = (
    U_recorded_full[train_mask], U_recorded_full[val_mask]
)
y_t_raw, y_v_raw = y_raw_full[train_mask], y_raw_full[val_mask]

# One shared scale is required because the decoder cumulatively sums
# one-step increments. A separate StandardScaler per horizon would make the
# cumulative increments dimensionally inconsistent across horizons.
delta_scale = float(np.std(y_t_raw))
if not np.isfinite(delta_scale) or delta_scale < 1e-6:
    delta_scale = 1.0
y_t = (y_t_raw / delta_scale).astype(np.float32)
y_v = (y_v_raw / delta_scale).astype(np.float32)
y_test = (y_raw_test / delta_scale).astype(np.float32)

sample_weights = np.ones(len(y_t), dtype=np.float32)
sample_weights[dynamic_full[train_mask]] = 4.0

train_dataset = TensorDataset(
    torch.from_numpy(X_t),
    torch.from_numpy(M_t),
    torch.from_numpy(U_recorded_t),
    torch.from_numpy(y_t),
    torch.from_numpy(sample_weights.reshape(-1, 1)),
)
val_dataset = TensorDataset(
    torch.from_numpy(X_v),
    torch.from_numpy(M_v),
    torch.from_numpy(U_recorded_v),
    torch.from_numpy(y_v),
)
test_dataset = TensorDataset(
    torch.from_numpy(X_test),
    torch.from_numpy(M_test),
    torch.from_numpy(U_recorded_test),
    torch.from_numpy(y_test),
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = FutureControlGRU(
    input_size=len(feature_cols),
    # 控制量之外，额外输入 future_control_known 标志。
    control_size=len(control_cols) + 1,
    hidden_size=HIDDEN_SIZE,
    time_steps=LOOK_BACK,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
)
horizon_weights = torch.tensor(
    HORIZON_WEIGHTS, dtype=torch.float32, device=device
)

best_val = float("inf")
best_state = None
wait = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []

    for xb, mb, ub_recorded, yb, wb in train_loader:
        xb = xb.to(device)
        mb = mb.to(device)
        ub_recorded = ub_recorded.to(device)
        yb, wb = yb.to(device), wb.to(device)
        if FUTURE_CONTROL_DROP_PROB > 0.0:
            ub_hold = build_hold_future_controls(
                xb, control_indices, MAX_HORIZON
            )
            ub = choose_future_controls(
                ub_recorded,
                ub_hold,
                FUTURE_CONTROL_DROP_PROB,
                training=True,
            )
        else:
            ub = ub_recorded
        optimizer.zero_grad(set_to_none=True)
        pred, rates, _ = model(xb, mb, ub)
        loss = compute_loss(pred, rates, yb, wb, horizon_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    val_hold_losses = []
    val_recorded_losses = []
    with torch.no_grad():
        for xb, mb, ub_recorded, yb in val_loader:
            xb = xb.to(device)
            mb = mb.to(device)
            ub_recorded = ub_recorded.to(device)
            yb = yb.to(device)
            ub_hold = build_hold_future_controls(
                xb, control_indices, MAX_HORIZON
            )
            pred_hold, rates_hold, _ = model(xb, mb, ub_hold)
            pred_recorded, rates_recorded, _ = model(
                xb, mb, ub_recorded
            )
            val_hold_losses.append(
                compute_loss(
                    pred_hold,
                    rates_hold,
                    yb,
                    torch.ones((len(yb), 1), device=device),
                    horizon_weights,
                ).item()
            )
            val_recorded_losses.append(
                compute_loss(
                    pred_recorded,
                    rates_recorded,
                    yb,
                    torch.ones((len(yb), 1), device=device),
                    horizon_weights,
                ).item()
            )

    train_loss = float(np.mean(train_losses))
    val_hold_loss = float(np.mean(val_hold_losses))
    val_recorded_loss = float(np.mean(val_recorded_losses))
    if PRIMARY_VALIDATION_MODE == "recorded":
        val_loss = val_recorded_loss
    elif PRIMARY_VALIDATION_MODE == "hold_last":
        val_loss = val_hold_loss
    else:
        raise ValueError(
            "PRIMARY_VALIDATION_MODE must be 'recorded' or 'hold_last'"
        )
    scheduler.step(val_loss)
    print(
        f"Epoch {epoch:03d}/{EPOCHS} - "
        f"train {train_loss:.6f} - "
        f"val_primary({PRIMARY_VALIDATION_MODE}) {val_loss:.6f} - "
        f"val_hold {val_hold_loss:.6f} - "
        f"val_recorded {val_recorded_loss:.6f}"
    )

    if val_loss < best_val:
        best_val = val_loss
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Early stopping")
            break

if best_state is not None:
    model.load_state_dict(best_state)

model.eval()
pred_scaled_hold = []
pred_scaled_recorded = []
with torch.no_grad():
    for xb, mb, ub_recorded, _ in test_loader:
        xb = xb.to(device)
        mb = mb.to(device)
        ub_recorded = ub_recorded.to(device)
        ub_hold = build_hold_future_controls(
            xb, control_indices, MAX_HORIZON
        )
        pred_hold, _, _ = model(xb, mb, ub_hold)
        pred_recorded, _, _ = model(xb, mb, ub_recorded)
        pred_scaled_hold.append(pred_hold.cpu().numpy())
        pred_scaled_recorded.append(pred_recorded.cpu().numpy())

pred_delta_raw_hold = np.vstack(pred_scaled_hold) * delta_scale
pred_delta_raw_recorded = np.vstack(pred_scaled_recorded) * delta_scale
pred_abs_hold = y_curr_test[:, None] + pred_delta_raw_hold
pred_abs_recorded = y_curr_test[:, None] + pred_delta_raw_recorded
real_abs = y_curr_test[:, None] + y_raw_test

mae_by_horizon_hold = np.mean(np.abs(pred_abs_hold - real_abs), axis=0)
mae_by_horizon_recorded = np.mean(
    np.abs(pred_abs_recorded - real_abs), axis=0
)
r2_by_horizon_hold = [
    r2_score(real_abs[:, j], pred_abs_hold[:, j])
    for j in range(len(HORIZONS))
]
r2_by_horizon_recorded = [
    r2_score(real_abs[:, j], pred_abs_recorded[:, j])
    for j in range(len(HORIZONS))
]
mae_persist = np.mean(
    np.abs(real_abs - y_curr_test[:, None]),
    axis=0,
)

print("\nTest results without future controls (hold-last deployment mode):")
for j, h in enumerate(HORIZONS):
    print(
        f"H={h:>3d}: MAE={mae_by_horizon_hold[j]:.4f} K, "
        f"R2={r2_by_horizon_hold[j]:.5f}, "
        f"persistence={mae_persist[j]:.4f} K"
    )

print("\nOracle diagnostic with recorded future valve controls:")
for j, h in enumerate(HORIZONS):
    print(
        f"H={h:>3d}: MAE={mae_by_horizon_recorded[j]:.4f} K, "
        f"R2={r2_by_horizon_recorded[j]:.5f}"
    )

region = (origin_test >= 8000) & (origin_test < 11000)
if np.any(region):
    print("\nFocused region: origin indices 8000:11000")
    for j, h in enumerate(HORIZONS):
        print(
            f"H={h:>3d}: "
            f"hold MAE={np.mean(np.abs(pred_abs_hold[region, j] - real_abs[region, j])):.4f} K, "
            f"oracle MAE={np.mean(np.abs(pred_abs_recorded[region, j] - real_abs[region, j])):.4f} K, "
            f"persistence={np.mean(np.abs(y_curr_test[region] - real_abs[region, j])):.4f} K"
        )

main_idx = HORIZONS.index(60)
plt.figure(figsize=(14, 8))
ax1 = plt.subplot(2, 1, 1)
ax1.plot(real_abs[:, main_idx], label="Actual T(t+60)", color="blue")
ax1.plot(
    pred_abs_recorded[:, main_idx],
    label="Predicted with recorded future controls",
    color="red",
    linestyle="--",
)
ax1.plot(
    pred_abs_hold[:, main_idx],
    label="Hold-last without future controls",
    color="gray",
    linestyle=":",
)
ax1.plot(y_curr_test, label="Persistence T(t)", color="gray", linestyle=":")
ax1.axvspan(
    8000, 11000, color="orange", alpha=0.12, label="Diagnostic region"
)
ax1.set_ylabel("Temperature [K]")
ax1.set_title(
    "Future-control GRU | paired by prediction origin "
    f"| recorded-control MAE={mae_by_horizon_recorded[main_idx]:.3f} K"
)
ax1.legend(loc="upper right")
ax1.grid(True, linestyle="--", alpha=0.5)

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
target_index = origin_test + HORIZONS[main_idx]
order = np.argsort(target_index)
actual_test_series = pd.to_numeric(
    df_test[TARGET_COL], errors="coerce"
).to_numpy(float)
ax2.plot(
    np.arange(len(actual_test_series)),
    actual_test_series,
    label="Raw test temperature",
    color="blue",
    alpha=0.45,
)
ax2.plot(
    target_index[order],
    pred_abs_recorded[order, main_idx],
    label="Recorded-control prediction at target t+60",
    color="red",
    linestyle="--",
)
ax2.plot(
    target_index[order],
    pred_abs_hold[order, main_idx],
    label="Hold-last prediction at target t+60",
    color="gray",
    linestyle=":",
)
ax2.set_ylabel("Temperature [K]")
ax2.legend(loc="upper right")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.set_yticks([])
ax2.set_xlabel("Target time index")
plt.tight_layout()
plot_path = ROOT / "future_control_multi_horizon_diagnostic.png"
plt.savefig(plot_path, dpi=200)
plt.close()
print("Saved plot:", plot_path)

model_path = SAVE_DIR / "future_control_multi_horizon_gru.pt"
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "model_type": "future_control_conditioned_multi_horizon_gru",
        "feature_cols": feature_cols,
        "control_cols": control_cols,
        "control_keys": CONTROL_KEYS,
        "look_back": LOOK_BACK,
        "horizons": HORIZONS,
        "hidden_size": HIDDEN_SIZE,
        "target_col": TARGET_COL,
        "mae_by_horizon_hold_last": mae_by_horizon_hold.tolist(),
        "r2_by_horizon_hold_last": r2_by_horizon_hold,
        "mae_by_horizon_recorded_oracle": mae_by_horizon_recorded.tolist(),
        "r2_by_horizon_recorded_oracle": r2_by_horizon_recorded,
        "mae_persistence": mae_persist.tolist(),
        "delta_scale": delta_scale,
        "future_controls_are_recorded_oracle": True,
        "future_control_drop_prob": FUTURE_CONTROL_DROP_PROB,
        "default_inference_mode": "recorded_or_mpc_planned",
        "primary_validation_mode": PRIMARY_VALIDATION_MODE,
        "future_control_known_channel": True,
    },
    model_path,
)
joblib.dump(scaler_x, SAVE_DIR / "scaler_x.pkl")
joblib.dump({"delta_scale": delta_scale}, SAVE_DIR / "delta_scaler.pkl")
joblib.dump(feature_cols, SAVE_DIR / "feature_cols.pkl")
joblib.dump(control_cols, SAVE_DIR / "control_cols.pkl")
joblib.dump(HORIZONS, SAVE_DIR / "horizons.pkl")
print("Saved model and scalers to:", SAVE_DIR)
