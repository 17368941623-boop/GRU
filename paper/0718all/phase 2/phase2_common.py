"""Leakage-safe Phase 2 architecture ablations for 60-step TE8353 forecasting.

Phase 1 is frozen at ``selected_plus_rebuilt_physics``.  Phase 2 changes only
the temporal pooling and decoder architecture.  Model selection uses the
complete, non-augmented Original 0501 validation curve and a strictly causal
subset whose 60-step input history contains a change in one of six valves.
The held-out test set is opt-in and must remain disabled during architecture
selection.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
PHASE1_DIR = ROOT.parent / "phase 1"
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))

from phase1_common import (  # noqa: E402
    DATA_DIR,
    DELTA_COL,
    LABEL_COL,
    LOOK_BACK,
    PREDICT_STEPS,
    SPECS as PHASE1_SPECS,
    TARGET_COL,
    WindowDataset,
    add_rebuilt_physics_features,
    add_target_delta,
    assert_group_separation,
    choose_device,
    compute_metrics,
    load_frame,
    resolve_feature_columns,
    save_curve,
    set_seed,
    validate_feature_schema,
)


OUTPUT_DIR = ROOT.parent / "outputs_phase2"
PHASE1_WINNER = "selected_plus_rebuilt_physics"
VALIDATION_FILE = "val_original_0501_full_clean.pkl"
CONTROL_KEYS = ("CV8300", "CV8313", "CV8310", "CV8311", "CV8312", "CV8351")
VALVE_EVENT_THRESHOLD = 0.05
FROZEN_ENGINEERED_FEATURES = {
    "CV8310_diff",
    "TE8353_TE8310_diff",
    "G_inlet_opening_proxy",
    "G_mix_conductance_proxy",
    "G_screen_conductance_proxy",
    "G_mix_thermal_drive",
    "A_8351_temperature_difference",
    "A_8351_flow_log_diff",
    "A_8351_transport_proxy",
}
LEGACY_EXCLUDED_FEATURES = {
    "CV8310_ratio",
    "Pseudo_Mass_Flow",
    "Pseudo_Cooling_Power",
    "Pseudo_Mass_Flow_8310",
    "Pseudo_Cooling_Power_8310",
    "Real_Mass_Flow_8351",
    "Actual_Cooling_Power_8351",
}


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    pooling: str  # "last", "attention", or "event_attention"
    decoder: str  # "direct" or "ode"


SPECS = {
    "gru_last": ArchitectureSpec("gru_last", "last", "direct"),
    "gru_attention": ArchitectureSpec("gru_attention", "attention", "direct"),
    "gru_event_attention": ArchitectureSpec("gru_event_attention", "event_attention", "direct"),
    "gru_ode": ArchitectureSpec("gru_ode", "last", "ode"),
    "gru_attention_ode": ArchitectureSpec("gru_attention_ode", "attention", "ode"),
    "gru_event_attention_ode": ArchitectureSpec(
        "gru_event_attention_ode", "event_attention", "ode"
    ),
}


def parse_args(spec: ArchitectureSpec) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Phase 2 architecture ablation: {spec.name}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--z-dim", type=int, default=32)
    parser.add_argument("--ode-hidden", type=int, default=64)
    parser.add_argument("--ode-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--test-file",
        default="test_clean.pkl",
        choices=("test_clean.pkl", "test_full_clean.pkl"),
        help="Used only with --evaluate-test after the architecture is frozen.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Opt in to held-out testing only after all Phase 2 choices are frozen.",
    )
    return parser.parse_args()


def resolve_raw_control_columns(frame: pd.DataFrame) -> list[str]:
    """Resolve exactly the six raw valve signals, never a derived proxy."""
    columns: list[str] = []
    for key in CONTROL_KEYS:
        pattern = re.compile(rf"^{re.escape(key)}(?:\s+ValueY)?$", flags=re.IGNORECASE)
        matches = [
            str(column)
            for column in frame.columns
            if pattern.fullmatch(str(column).strip())
        ]
        if len(matches) != 1:
            raise KeyError(f"Expected one raw column for control {key}, found {matches}")
        columns.append(matches[0])
    return columns


def add_six_valve_event_mask(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a causal event flag from current-versus-previous raw valve values."""
    if "file_id" not in frame.columns:
        raise KeyError("file_id is required to prevent valve differences across files")
    frame = frame.copy()
    event = np.zeros(len(frame), dtype=bool)
    for column in resolve_raw_control_columns(frame):
        difference = frame.groupby("file_id", sort=False)[column].diff().abs()
        event |= difference.gt(VALVE_EVENT_THRESHOLD).fillna(False).to_numpy()
    frame["valve_event_mask"] = event.astype(np.float32)
    return frame


def prepare_frame(filename: str) -> pd.DataFrame:
    frame = add_six_valve_event_mask(load_frame(filename))
    return add_target_delta(add_rebuilt_physics_features(frame))


def frozen_feature_columns(train_frame: pd.DataFrame) -> list[str]:
    features = resolve_feature_columns(train_frame, PHASE1_SPECS[PHASE1_WINNER])
    missing = FROZEN_ENGINEERED_FEATURES - set(features)
    legacy_present = LEGACY_EXCLUDED_FEATURES & set(features)
    if missing:
        raise ValueError(f"Frozen Phase 1 engineered features are missing: {sorted(missing)}")
    if legacy_present:
        raise ValueError(f"Legacy proxy features unexpectedly entered Phase 2: {sorted(legacy_present)}")
    if len(features) != 37:
        raise ValueError(f"Phase 2 expected 37 frozen inputs, found {len(features)}")
    return features


def find_control_indices(features: list[str]) -> tuple[list[str], list[int]]:
    columns: list[str] = []
    for key in CONTROL_KEYS:
        pattern = re.compile(rf"^{re.escape(key)}(?:\s+ValueY)?$", flags=re.IGNORECASE)
        matches = [column for column in features if pattern.fullmatch(str(column).strip())]
        if len(matches) != 1:
            raise KeyError(f"Expected one frozen input column for control {key}, found {matches}")
        columns.append(matches[0])
    return columns, [features.index(column) for column in columns]


class Phase2WindowDataset(Dataset):
    """Fixed-horizon windows with a causal historical event mask."""

    def __init__(self, frame: pd.DataFrame, x_scaled: np.ndarray, y_scaled: np.ndarray):
        if "valve_event_mask" not in frame.columns:
            raise KeyError("valve_event_mask is required by the Phase 2 attention ablation")
        self.x = np.asarray(x_scaled, dtype=np.float32)
        self.y = np.asarray(y_scaled, dtype=np.float32).reshape(-1)
        self.current = frame[TARGET_COL].to_numpy(dtype=np.float32)
        self.future = frame[LABEL_COL].to_numpy(dtype=np.float32)
        self.event = frame["valve_event_mask"].to_numpy(dtype=np.float32)
        self.ends = WindowDataset._build_window_ends(frame)
        if len(self.ends) == 0:
            raise ValueError("No valid Phase 2 windows")

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, index: int):
        end = int(self.ends[index])
        start = end - LOOK_BACK + 1
        return (
            torch.from_numpy(self.x[start : end + 1]),
            torch.from_numpy(self.event[start : end + 1, None]),
            torch.tensor(self.y[end], dtype=torch.float32),
            torch.tensor(self.current[end], dtype=torch.float32),
            torch.tensor(self.future[end], dtype=torch.float32),
        )


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int, time_steps: int, event_prior: bool):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(time_steps, 1))
        self.event_prior = event_prior
        if event_prior:
            self.event_boost_raw = nn.Parameter(torch.tensor(-2.0))

    def forward(self, encoded: torch.Tensor, event_mask: torch.Tensor):
        logits = torch.tanh(self.score(encoded) + self.position_bias.unsqueeze(0))
        if self.event_prior:
            # Softplus enforces a non-negative prior boost while keeping its
            # magnitude learnable from training data only.
            logits = logits + event_mask * F.softplus(self.event_boost_raw)
        alpha = torch.softmax(logits.squeeze(-1), dim=1)
        context = torch.sum(encoded * alpha.unsqueeze(-1), dim=1)
        return context, alpha


class ODEFunc(nn.Module):
    def __init__(self, z_dim: int, control_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + control_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, z: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, control], dim=-1))


def rk4_integrate(
    func: ODEFunc,
    z0: torch.Tensor,
    control: torch.Tensor,
    steps: int,
    horizon: float = 1.0,
) -> torch.Tensor:
    z = z0
    step_size = horizon / steps
    for _ in range(steps):
        k1 = func(z, control)
        k2 = func(z + 0.5 * step_size * k1, control)
        k3 = func(z + 0.5 * step_size * k2, control)
        k4 = func(z + step_size * k3, control)
        z = z + step_size * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return z


class Phase2Model(nn.Module):
    """Common GRU encoder with controlled pooling/decoder ablations."""

    def __init__(
        self,
        input_size: int,
        control_indices: list[int],
        spec: ArchitectureSpec,
        hidden_size: int,
        dropout: float,
        z_dim: int,
        ode_hidden: int,
        ode_steps: int,
    ):
        super().__init__()
        self.spec = spec
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.register_buffer("control_indices", torch.tensor(control_indices, dtype=torch.long))

        if spec.pooling == "last":
            self.attention = None
        elif spec.pooling in {"attention", "event_attention"}:
            self.attention = TemporalAttention(
                hidden_size,
                LOOK_BACK,
                event_prior=spec.pooling == "event_attention",
            )
        else:
            raise ValueError(f"Unknown pooling mode: {spec.pooling}")

        if spec.decoder == "direct":
            self.direct_head = nn.Sequential(
                nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1)
            )
            self.to_z0 = self.ode_func = self.ode_head = None
        elif spec.decoder == "ode":
            self.direct_head = None
            self.to_z0 = nn.Linear(hidden_size, z_dim)
            self.ode_func = ODEFunc(z_dim, len(control_indices), ode_hidden)
            self.ode_head = nn.Sequential(nn.Linear(z_dim, 32), nn.ReLU(), nn.Linear(32, 1))
        else:
            raise ValueError(f"Unknown decoder mode: {spec.decoder}")
        self.ode_steps = ode_steps

    def forward(self, x: torch.Tensor, event_mask: torch.Tensor):
        encoded, _ = self.gru1(x)
        encoded = self.dropout(encoded)
        encoded, _ = self.gru2(encoded)
        if self.attention is None:
            context = encoded[:, -1, :]
            alpha = None
        else:
            context, alpha = self.attention(encoded, event_mask)

        if self.spec.decoder == "direct":
            assert self.direct_head is not None
            prediction = self.direct_head(context)
        else:
            assert self.to_z0 is not None and self.ode_func is not None and self.ode_head is not None
            z0 = self.to_z0(context)
            # No future control is read.  The six current valve openings are
            # held constant over normalized latent time for this Phase 2 test.
            current_control = x[:, -1, self.control_indices]
            z_horizon = rk4_integrate(self.ode_func, z0, current_control, self.ode_steps)
            prediction = self.ode_head(z_horizon)
        return prediction.squeeze(-1), alpha


def make_loader(dataset: Dataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_model(
    model: Phase2Model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    criterion = nn.HuberLoss(delta=1.0)
    best_selection_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, event_mask, target, _, _ in train_loader:
            x = x.to(device)
            event_mask = event_mask.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(x, event_mask)
            loss = criterion(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_event_loss_sum = 0.0
        val_event_count = 0
        with torch.no_grad():
            for x, event_mask, target, _, _ in val_loader:
                x = x.to(device)
                event_mask = event_mask.to(device)
                target = target.to(device)
                prediction, _ = model(x, event_mask)
                per_window_loss = F.huber_loss(
                    prediction, target, delta=1.0, reduction="none"
                )
                val_loss_sum += float(per_window_loss.sum().item())
                val_count += int(len(per_window_loss))
                historical_event = event_mask.squeeze(-1).sum(dim=1) > 0
                if torch.any(historical_event):
                    val_event_loss_sum += float(per_window_loss[historical_event].sum().item())
                    val_event_count += int(historical_event.sum().item())

        train_loss = float(np.mean(train_losses))
        if val_count == 0 or val_event_count == 0:
            raise RuntimeError(
                "Original validation curve produced no windows or no causal historical-event windows"
            )
        val_all_loss = val_loss_sum / val_count
        val_event_loss = val_event_loss_sum / val_event_count
        selection_loss = val_event_loss
        scheduler.step(selection_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_huber": train_loss,
                "val_huber": selection_loss,
                "val_huber_all_windows": val_all_loss,
                "val_huber_history_event": val_event_loss,
                "val_history_event_windows": val_event_count,
                "learning_rate": learning_rate,
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs}: train_huber={train_loss:.6f}, "
            f"val_all={val_all_loss:.6f}, val_history_event={val_event_loss:.6f}, "
            f"lr={learning_rate:.2e}"
        )

        if selection_loss < best_selection_loss:
            best_selection_loss = selection_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(
                    f"Early stopping at epoch {epoch}; "
                    f"best causal history-event Huber={best_selection_loss:.6f}"
                )
                break

    if best_state is None:
        raise RuntimeError("Phase 2 training did not produce a checkpoint")
    return best_state, history, best_selection_loss


def predict_dataset(
    model: Phase2Model,
    loader: DataLoader,
    device: torch.device,
    scaler_y: StandardScaler,
):
    pred_scaled, current, future, historical_event = [], [], [], []
    attention_sum = np.zeros(LOOK_BACK, dtype=np.float64)
    attention_count = 0
    event_mass_sum = 0.0
    event_present_mass_sum = 0.0
    event_present_count = 0

    model.eval()
    with torch.no_grad():
        for x, event_mask, _, current_temp, future_temp in loader:
            prediction, alpha = model(x.to(device), event_mask.to(device))
            pred_scaled.append(prediction.cpu().numpy())
            current.append(current_temp.numpy())
            future.append(future_temp.numpy())
            historical_event.append(
                (event_mask.squeeze(-1).sum(dim=1) > 0).numpy()
            )
            if alpha is not None:
                alpha_np = alpha.cpu().numpy()
                mask_np = event_mask.squeeze(-1).numpy()
                attention_sum += alpha_np.sum(axis=0)
                attention_count += len(alpha_np)
                event_mass = np.sum(alpha_np * mask_np, axis=1)
                event_mass_sum += float(event_mass.sum())
                present = np.sum(mask_np, axis=1) > 0
                if np.any(present):
                    event_present_mass_sum += float(event_mass[present].sum())
                    event_present_count += int(present.sum())

    delta = scaler_y.inverse_transform(np.concatenate(pred_scaled).reshape(-1, 1)).reshape(-1)
    current_arr = np.concatenate(current)
    future_arr = np.concatenate(future)
    prediction_arr = current_arr + delta
    attention_diagnostics = None
    if attention_count:
        attention_diagnostics = {
            "mean_attention_by_position": (attention_sum / attention_count).tolist(),
            "mean_event_attention_mass_all_windows": event_mass_sum / attention_count,
            "mean_event_attention_mass_event_windows": (
                event_present_mass_sum / event_present_count if event_present_count else 0.0
            ),
            "event_window_count": event_present_count,
        }
    return (
        prediction_arr,
        future_arr,
        current_arr,
        delta,
        attention_diagnostics,
        np.concatenate(historical_event).astype(bool),
    )


def subset_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    current: np.ndarray,
    mask: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    if not np.any(mask):
        raise ValueError("Requested evaluation subset is empty")
    return compute_metrics(actual[mask], predicted[mask]), compute_metrics(
        actual[mask], current[mask]
    )


def save_attention_diagnostics(diagnostics: dict | None, path: Path) -> None:
    if diagnostics is None:
        return
    weights = diagnostics["mean_attention_by_position"]
    pd.DataFrame(
        {
            "history_offset": np.arange(-LOOK_BACK + 1, 1),
            "mean_attention_weight": weights,
        }
    ).to_csv(path, index=False)


def run_experiment(spec_name: str) -> None:
    if spec_name not in SPECS:
        raise KeyError(f"Unknown Phase 2 experiment: {spec_name}")
    spec = SPECS[spec_name]
    args = parse_args(spec)
    set_seed(args.seed)
    device = choose_device()
    print(f"Experiment={spec.name}; device={device}; data_dir={DATA_DIR}")

    train_frame = prepare_frame("train_clean.pkl")
    val_frame = prepare_frame(VALIDATION_FILE)
    assert_group_separation(train_frame, val_frame)
    features = frozen_feature_columns(train_frame)
    control_columns, control_indices = find_control_indices(features)
    validate_feature_schema(train_frame, features, "train")
    validate_feature_schema(val_frame, features, "validation")

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    train_x = scaler_x.fit_transform(train_frame[features]).astype(np.float32)
    train_y = scaler_y.fit_transform(train_frame[[DELTA_COL]]).astype(np.float32)
    val_x = scaler_x.transform(val_frame[features]).astype(np.float32)
    val_y = scaler_y.transform(val_frame[[DELTA_COL]]).astype(np.float32)

    train_dataset = Phase2WindowDataset(train_frame, train_x, train_y)
    val_dataset = Phase2WindowDataset(val_frame, val_x, val_y)
    train_loader = make_loader(train_dataset, args, shuffle=True)
    val_loader = make_loader(val_dataset, args, shuffle=False)
    model = Phase2Model(
        input_size=len(features),
        control_indices=control_indices,
        spec=spec,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        z_dim=args.z_dim,
        ode_hidden=args.ode_hidden,
        ode_steps=args.ode_steps,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Frozen features={len(features)}; controls={control_columns}; "
        f"parameters={parameter_count:,}"
    )

    best_state, history, best_selection_huber = train_model(
        model, train_loader, val_loader, args, device
    )
    model.load_state_dict(best_state)
    (
        val_prediction,
        val_actual,
        val_current,
        val_delta,
        val_attention,
        val_history_event,
    ) = predict_dataset(model, val_loader, device, scaler_y)
    val_metrics = compute_metrics(val_actual, val_prediction)
    val_persistence_metrics = compute_metrics(val_actual, val_current)
    val_event_metrics, val_event_persistence_metrics = subset_metrics(
        val_actual, val_prediction, val_current, val_history_event
    )

    test_frame: pd.DataFrame | None = None
    test_metrics: dict[str, float] | None = None
    test_outputs = None
    test_attention = None
    test_event_metrics = None
    test_event_persistence_metrics = None
    test_persistence_metrics = None
    if args.evaluate_test:
        test_frame = prepare_frame(args.test_file)
        assert_group_separation(train_frame, val_frame, test_frame)
        validate_feature_schema(test_frame, features, "test")
        test_x = scaler_x.transform(test_frame[features]).astype(np.float32)
        test_y = scaler_y.transform(test_frame[[DELTA_COL]]).astype(np.float32)
        test_dataset = Phase2WindowDataset(test_frame, test_x, test_y)
        test_loader = make_loader(test_dataset, args, shuffle=False)
        test_outputs = predict_dataset(model, test_loader, device, scaler_y)
        (
            test_prediction,
            test_actual,
            test_current,
            _,
            test_attention,
            test_history_event,
        ) = test_outputs
        test_metrics = compute_metrics(test_actual, test_prediction)
        test_persistence_metrics = compute_metrics(test_actual, test_current)
        test_event_metrics, test_event_persistence_metrics = subset_metrics(
            test_actual, test_prediction, test_current, test_history_event
        )

    evaluation_scope = (
        f"final_{Path(args.test_file).stem}" if args.evaluate_test else "validation_only"
    )
    run_dir = OUTPUT_DIR / spec.name / evaluation_scope / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "phase": 2,
        "experiment": spec.name,
        "pooling": spec.pooling,
        "decoder": spec.decoder,
        "phase1_frozen_experiment": PHASE1_WINNER,
        "seed": args.seed,
        "look_back": LOOK_BACK,
        "predict_steps": PREDICT_STEPS,
        "target": TARGET_COL,
        "parameter_count": parameter_count,
        "feature_count": len(features),
        "feature_columns": features,
        "ode_control_assumption": "hold_last_current_valves" if spec.decoder == "ode" else None,
        "control_columns": control_columns if spec.decoder == "ode" else [],
        "test_evaluated": args.evaluate_test,
        "test_file": args.test_file if args.evaluate_test else None,
        "validation_file": VALIDATION_FILE,
        "validation_protocol": "complete_original_0501_non_augmented",
        "selection_metric": "validation_history_event_huber_scaled_delta",
        "best_selection_huber": best_selection_huber,
        "valve_event_controls": list(CONTROL_KEYS),
        "valve_event_threshold": VALVE_EVENT_THRESHOLD,
        "train_source_groups": sorted(train_frame["source_group"].unique().tolist()),
        "validation_source_groups": sorted(val_frame["source_group"].unique().tolist()),
        "test_source_groups": (
            sorted(test_frame["source_group"].unique().tolist()) if test_frame is not None else []
        ),
        "validation_metrics": val_metrics,
        "validation_persistence_metrics": val_persistence_metrics,
        "validation_history_event_metrics": val_event_metrics,
        "validation_history_event_persistence_metrics": val_event_persistence_metrics,
        "validation_history_event_windows": int(val_history_event.sum()),
        "test_metrics": test_metrics,
        "test_persistence_metrics": test_persistence_metrics,
        "test_history_event_metrics": test_event_metrics,
        "test_history_event_persistence_metrics": test_event_persistence_metrics,
        "validation_attention": val_attention,
        "test_attention": test_attention,
        "training_rows": int(len(train_frame)),
        "validation_rows": int(len(val_frame)),
        "test_rows": int(len(test_frame)) if test_frame is not None else 0,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save({"model_state_dict": model.state_dict(), **metadata}, run_dir / "model.pt")
    joblib.dump(scaler_x, run_dir / "scaler_x.pkl")
    joblib.dump(scaler_y, run_dir / "scaler_y.pkl")
    pd.DataFrame(
        {
            "origin_row": val_dataset.ends,
            "current_temperature_k": val_current,
            "actual_temperature_k": val_actual,
            "predicted_temperature_k": val_prediction,
            "predicted_delta_k": val_delta,
            "true_delta_k": val_actual - val_current,
            "historical_six_valve_event": val_history_event.astype(int),
            "model_absolute_error_k": np.abs(val_prediction - val_actual),
            "persistence_absolute_error_k": np.abs(val_current - val_actual),
        }
    ).to_csv(run_dir / "val_predictions.csv", index=False)
    save_curve(val_actual, val_prediction, run_dir / "val_prediction.png", f"{spec.name} | validation")
    save_attention_diagnostics(val_attention, run_dir / "val_attention.csv")

    if test_outputs is not None:
        (
            test_prediction,
            test_actual,
            test_current,
            test_delta,
            _,
            test_history_event,
        ) = test_outputs
        pd.DataFrame(
            {
                "origin_row": test_dataset.ends,
                "current_temperature_k": test_current,
                "actual_temperature_k": test_actual,
                "predicted_temperature_k": test_prediction,
                "predicted_delta_k": test_delta,
                "true_delta_k": test_actual - test_current,
                "historical_six_valve_event": test_history_event.astype(int),
                "model_absolute_error_k": np.abs(test_prediction - test_actual),
                "persistence_absolute_error_k": np.abs(test_current - test_actual),
            }
        ).to_csv(run_dir / "test_predictions.csv", index=False)
        save_curve(
            test_actual,
            test_prediction,
            run_dir / "test_prediction.png",
            f"{spec.name} | {args.test_file}",
        )
        save_attention_diagnostics(test_attention, run_dir / "test_attention.csv")

    print(
        json.dumps(
            {
                "validation_overall": val_metrics,
                "validation_persistence": val_persistence_metrics,
                "validation_history_event": val_event_metrics,
                "validation_history_event_persistence": val_event_persistence_metrics,
                "test": test_metrics,
            },
            indent=2,
        )
    )
    print(f"Saved to {run_dir}")
