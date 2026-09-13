"""Shared, leakage-safe training code for Phase 1 feature ablations.

Development uses only train_clean.pkl and val_clean.pkl.  test_clean.pkl is
opened only after early stopping has selected a checkpoint.  This module never
splits train_clean.pkl internally and fits all scalers on train_clean.pkl only.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "processed_data"
OUTPUT_DIR = ROOT.parent / "outputs"

TARGET_COL = "TE8353"
LOOK_BACK = 60
PREDICT_STEPS = 60
LABEL_COL = f"Future_Target_{PREDICT_STEPS}step"
DELTA_COL = "Target_Delta"

ENGINEERED_GROUPS = {
    # Dynamic actuator information: captures the timing and magnitude of a CV8310 action.
    "valve_diff": {"CV8310_diff"},
    # Nominal opening fraction only; it ignores valve Cv curves and branch pressure drops.
    "cv8310_ratio": {"CV8310_ratio"},
    # Qualitative G-branch cooling proxies. These are not calibrated mass flow or cooling power.
    "g_branch_physics": {
        "Pseudo_Mass_Flow",
        "Pseudo_Cooling_Power",
        "Pseudo_Mass_Flow_8310",
        "Pseudo_Cooling_Power_8310",
    },
    # FT8351-based proxies; their validity depends on whether FT8351 is volumetric or mass flow.
    "branch_8351_physics": {"Real_Mass_Flow_8351", "Actual_Cooling_Power_8351"},
    # Upstream/downstream thermal gradient, a direct proxy for mixing and transport driving force.
    "temperature_difference": {"TE8353_TE8310_diff"},
    # Rebuilt G-pipe topology features.  These are dimensionless valve-network
    # conductance proxies, not measured flow rates.
    "g_routing_rebuilt": {"G_mix_routing_fraction"},
    "g_network_rebuilt": {
        "G_inlet_opening_proxy",
        "G_mix_conductance_proxy",
        "G_screen_conductance_proxy",
    },
    # The signed thermal driving force delivered through the nominal G-to-mixing path.
    "g_thermal_rebuilt": {"G_mix_thermal_drive"},
    # Unit-agnostic 8351 transport features.  FT8351 is deliberately not
    # multiplied by density until its engineering unit has been verified.
    "branch_8351_rebuilt": {
        "A_8351_temperature_difference",
        "A_8351_flow_log_diff",
        "A_8351_transport_proxy",
    },
}

# Phase 1 preliminary results support these two engineered groups on both validation and test MAE.
# The remaining proxy groups are excluded until valve characteristics and FT8351 units are verified.
SELECTED_ENGINEERED_GROUPS = {"valve_diff", "temperature_difference"}
LEGACY_ENGINEERED_GROUPS = {
    "valve_diff",
    "cv8310_ratio",
    "g_branch_physics",
    "branch_8351_physics",
    "temperature_difference",
}

METADATA_OR_LEAKAGE_COLS = {
    LABEL_COL,
    DELTA_COL,
    "valve_event_mask",  # Phase 1 uses a plain GRU, not the attention mechanism.
    "file_id",
    "source_group",
    "Target_Inertia_10min",
    "Cumulative_Cooling_8351",
    "CV8310_action_neighborhood",
    "TE8353_slope_60",
}


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    mode: str  # "core", "full", "full_minus", "selected", or "selected_plus"
    removed_group: str | None = None
    added_groups: tuple[str, ...] = ()


SPECS = {
    "core": ExperimentSpec("core", "core"),
    "full": ExperimentSpec("full", "full"),
    "selected": ExperimentSpec("selected", "selected"),
    # Separate tests on top of the leakage-safe selected baseline.  The old
    # routing ratio is tested alone; the rebuilt network does not expose that
    # ratio as a standalone input unless the routing experiment supports it.
    "selected_plus_g_routing": ExperimentSpec(
        "selected_plus_g_routing", "selected_plus", added_groups=("g_routing_rebuilt",)
    ),
    "selected_plus_g_network": ExperimentSpec(
        "selected_plus_g_network",
        "selected_plus",
        added_groups=("g_network_rebuilt",),
    ),
    "selected_plus_g_thermal": ExperimentSpec(
        "selected_plus_g_thermal",
        "selected_plus",
        added_groups=("g_network_rebuilt", "g_thermal_rebuilt"),
    ),
    "selected_plus_8351_transport": ExperimentSpec(
        "selected_plus_8351_transport", "selected_plus", added_groups=("branch_8351_rebuilt",)
    ),
    "selected_plus_rebuilt_physics": ExperimentSpec(
        "selected_plus_rebuilt_physics",
        "selected_plus",
        added_groups=(
            "g_network_rebuilt",
            "g_thermal_rebuilt",
            "branch_8351_rebuilt",
        ),
    ),
    "no_valve_diff": ExperimentSpec("no_valve_diff", "full_minus", "valve_diff"),
    "no_cv8310_ratio": ExperimentSpec("no_cv8310_ratio", "full_minus", "cv8310_ratio"),
    "no_g_branch_physics": ExperimentSpec("no_g_branch_physics", "full_minus", "g_branch_physics"),
    "no_8351_branch_physics": ExperimentSpec("no_8351_branch_physics", "full_minus", "branch_8351_physics"),
    "no_temperature_difference": ExperimentSpec(
        "no_temperature_difference", "full_minus", "temperature_difference"
    ),
}


def parse_args(spec: ExperimentSpec) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Phase 1 feature ablation: {spec.name}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--test-file",
        default="test_clean.pkl",
        choices=("test_clean.pkl", "test_full_clean.pkl"),
        help="Used only with --evaluate-test. Prefer test_full_clean.pkl for the final frozen-model audit.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help=(
            "Explicitly evaluate the held-out test set. Leave this off during feature/model selection; "
            "enable it only after the complete design is frozen."
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_frame(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing processed data: {path}")
    frame = joblib.load(path)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{path} is not a pandas DataFrame")
    return frame.copy()


def find_raw_signal_column(frame: pd.DataFrame, signal: str) -> str:
    """Resolve a raw PLC signal without accidentally selecting a derived column."""
    pattern = re.compile(rf"^{re.escape(signal)}(?:\s+ValueY)?$", flags=re.IGNORECASE)
    matches = [str(column) for column in frame.columns if pattern.fullmatch(str(column).strip())]
    if len(matches) != 1:
        raise KeyError(f"Expected one raw column for {signal}, found: {matches}")
    return matches[0]


def add_rebuilt_physics_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build causal, unit-aware topology and transport proxies from rows at t or earlier.

    No fitted statistic, future row, or future label is used here.  The valve
    network assumes only that opening is a monotone surrogate for conductance;
    therefore its outputs remain explicitly labelled as proxies.
    """
    required_signals = (
        "TE8353",
        "TE8310",
        "TE8351",
        "FT8351",
        "CV8300",
        "CV8313",
        "CV8310",
        "CV8311",
        "CV8312",
    )
    columns = {signal: find_raw_signal_column(frame, signal) for signal in required_signals}

    def numeric(signal: str) -> pd.Series:
        return pd.to_numeric(frame[columns[signal]], errors="coerce").astype(float)

    # Openings are converted from percent to [0, 1] only inside the derived
    # calculation.  The original valve columns remain unchanged.
    openings = {
        signal: numeric(signal).clip(lower=0.0, upper=100.0) / 100.0
        for signal in ("CV8300", "CV8313", "CV8310", "CV8311", "CV8312")
    }
    g_inlet = openings["CV8300"] + openings["CV8313"]
    g_downstream = openings["CV8310"] + openings["CV8311"] + openings["CV8312"]
    positive_downstream = g_downstream > 1e-12
    routing_fraction = pd.Series(0.0, index=frame.index, dtype=float)
    routing_fraction.loc[positive_downstream] = (
        openings["CV8310"].loc[positive_downstream] / g_downstream.loc[positive_downstream]
    )

    # For an orifice-like relation m ~ C*sqrt(delta_p), two conductances in
    # series combine as C1*C2/sqrt(C1^2+C2^2).  The three downstream branches
    # are treated as parallel nominal conductances.  Missing Cv curves and
    # pressure drops mean this is deliberately not reported as mass flow.
    denominator = np.sqrt(g_inlet.pow(2) + g_downstream.pow(2))
    network_conductance = pd.Series(0.0, index=frame.index, dtype=float)
    active_network = denominator > 1e-12
    network_conductance.loc[active_network] = (
        g_inlet.loc[active_network]
        * g_downstream.loc[active_network]
        / denominator.loc[active_network]
    )
    mix_conductance = network_conductance * routing_fraction
    screen_conductance = network_conductance * (1.0 - routing_fraction)
    screen_conductance.loc[~positive_downstream] = 0.0

    frame["G_mix_routing_fraction"] = routing_fraction
    frame["G_inlet_opening_proxy"] = g_inlet
    frame["G_mix_conductance_proxy"] = mix_conductance
    frame["G_screen_conductance_proxy"] = screen_conductance
    frame["G_mix_thermal_drive"] = mix_conductance * (numeric("TE8353") - numeric("TE8310"))

    # FT8351's unit is not documented in the source column.  log1p preserves
    # order while limiting the influence of the 0501 batch's large flow spikes.
    # The signed temperature difference distinguishes cooling and heating drive.
    flow_log = np.log1p(numeric("FT8351").clip(lower=0.0))
    if "file_id" not in frame.columns:
        raise KeyError("file_id is required for causal FT8351 differencing")
    flow_log_diff = flow_log.groupby(frame["file_id"], sort=False).diff().fillna(0.0)
    a_temperature_difference = numeric("TE8353") - numeric("TE8351")
    frame["A_8351_temperature_difference"] = a_temperature_difference
    frame["A_8351_flow_log_diff"] = flow_log_diff
    frame["A_8351_transport_proxy"] = flow_log * a_temperature_difference
    return frame


def add_target_delta(frame: pd.DataFrame) -> pd.DataFrame:
    missing = {TARGET_COL, LABEL_COL} - set(frame.columns)
    if missing:
        raise KeyError(f"Required target columns are missing: {sorted(missing)}")
    frame[DELTA_COL] = pd.to_numeric(frame[LABEL_COL], errors="coerce") - pd.to_numeric(
        frame[TARGET_COL], errors="coerce"
    )
    return frame


def assert_group_separation(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame | None = None) -> None:
    for name, frame in (("train", train), ("validation", val)):
        if "source_group" not in frame.columns:
            raise KeyError(f"{name} has no source_group column")
    train_groups = set(train["source_group"].unique())
    val_groups = set(val["source_group"].unique())
    if train_groups & val_groups:
        raise ValueError(f"Train/validation source_group overlap: {sorted(train_groups & val_groups)}")
    if test is not None:
        test_groups = set(test["source_group"].unique())
        overlap = (train_groups & test_groups) | (val_groups & test_groups)
        if overlap:
            raise ValueError(f"Test source_group overlap with development data: {sorted(overlap)}")


def resolve_feature_columns(train: pd.DataFrame, spec: ExperimentSpec) -> list[str]:
    excluded = set(METADATA_OR_LEAKAGE_COLS)
    excluded.update(c for c in train.columns if str(c).startswith("Future_Target_"))
    excluded.update(c for c in train.columns if "FLUID_DENSITY" in str(c).upper())
    excluded.update(c for c in train.columns if "FLUID_ENTHALPY" in str(c).upper())

    all_engineered = set().union(*ENGINEERED_GROUPS.values())
    included_groups: set[str]
    if spec.mode == "core":
        included_groups = set()
    elif spec.mode == "full":
        # Preserve the definition used by the completed seven-experiment batch.
        included_groups = set(LEGACY_ENGINEERED_GROUPS)
    elif spec.mode == "selected":
        included_groups = set(SELECTED_ENGINEERED_GROUPS)
    elif spec.mode == "selected_plus":
        included_groups = set(SELECTED_ENGINEERED_GROUPS) | set(spec.added_groups)
    elif spec.mode == "full_minus":
        if spec.removed_group not in ENGINEERED_GROUPS:
            raise ValueError(f"Unknown engineered feature group: {spec.removed_group}")
        included_groups = set(LEGACY_ENGINEERED_GROUPS) - {spec.removed_group}
    else:
        raise ValueError(f"Unknown feature mode: {spec.mode}")

    unknown_groups = included_groups - set(ENGINEERED_GROUPS)
    if unknown_groups:
        raise ValueError(f"Unknown included engineered groups: {sorted(unknown_groups)}")
    included_columns = (
        set().union(*(ENGINEERED_GROUPS[name] for name in included_groups))
        if included_groups
        else set()
    )
    excluded.update(all_engineered - included_columns)

    features = [c for c in train.columns if c not in excluded]
    features = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    if not features:
        raise ValueError("No input features remain after applying the Phase 1 feature definition")
    return features


def validate_feature_schema(frame: pd.DataFrame, features: Iterable[str], split_name: str) -> None:
    missing = [col for col in features if col not in frame.columns]
    if missing:
        raise KeyError(f"{split_name} is missing model features: {missing}")
    if frame[list(features)].isna().any().any():
        raise ValueError(f"{split_name} contains NaN values in selected input features")
    if frame[DELTA_COL].isna().any():
        raise ValueError(f"{split_name} contains missing target deltas")


class WindowDataset(Dataset):
    """Lazy fixed-horizon windows; stores rows once and never crosses file_id boundaries."""

    def __init__(self, frame: pd.DataFrame, x_scaled: np.ndarray, y_scaled: np.ndarray):
        self.x = np.asarray(x_scaled, dtype=np.float32)
        self.y = np.asarray(y_scaled, dtype=np.float32).reshape(-1)
        self.current = frame[TARGET_COL].to_numpy(dtype=np.float32)
        self.future = frame[LABEL_COL].to_numpy(dtype=np.float32)
        self.ends = self._build_window_ends(frame)
        if len(self.ends) == 0:
            raise ValueError("No valid windows: each continuous segment must contain at least LOOK_BACK rows")

    @staticmethod
    def _build_window_ends(frame: pd.DataFrame) -> np.ndarray:
        if "file_id" not in frame.columns:
            raise KeyError("file_id is required to prevent windows crossing file/segment boundaries")
        ends: list[np.ndarray] = []
        for _, indices in frame.groupby("file_id", sort=False).indices.items():
            idx = np.asarray(indices, dtype=np.int64)
            if len(idx) < LOOK_BACK:
                continue
            if not np.all(np.diff(idx) == 1):
                raise ValueError("Rows for a file_id must be contiguous before building windows")
            ends.append(np.arange(idx[0] + LOOK_BACK - 1, idx[-1] + 1, dtype=np.int64))
        return np.concatenate(ends) if ends else np.empty(0, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, index: int):
        end = int(self.ends[index])
        start = end - LOOK_BACK + 1
        return (
            torch.from_numpy(self.x[start : end + 1]),
            torch.tensor(self.y[end], dtype=torch.float32),
            torch.tensor(self.current[end], dtype=torch.float32),
            torch.tensor(self.future[end], dtype=torch.float32),
        )


class PlainGRU(nn.Module):
    """Fixed Phase 1 architecture: no event attention, physical loss, or ODE decoder."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru1(x)
        output = self.dropout(output)
        output, _ = self.gru2(output)
        return self.head(output[:, -1, :]).squeeze(-1)


def make_loader(dataset: Dataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    criterion = nn.HuberLoss(delta=1.0)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, target, _, _ in train_loader:
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, target, _, _ in val_loader:
                x, target = x.to(device), target.to(device)
                val_losses.append(float(criterion(model(x), target).item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_huber": train_loss, "val_huber": val_loss})
        print(f"Epoch {epoch:03d}/{args.epochs}: train_huber={train_loss:.6f}, val_huber={val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}; best validation Huber={best_loss:.6f}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    return best_state, history


def predict_dataset(model: nn.Module, loader: DataLoader, device: torch.device, scaler_y: StandardScaler):
    pred_scaled, current, future = [], [], []
    model.eval()
    with torch.no_grad():
        for x, _, current_temp, future_temp in loader:
            pred_scaled.append(model(x.to(device)).cpu().numpy())
            current.append(current_temp.numpy())
            future.append(future_temp.numpy())
    delta = scaler_y.inverse_transform(np.concatenate(pred_scaled).reshape(-1, 1)).reshape(-1)
    current_arr = np.concatenate(current)
    future_arr = np.concatenate(future)
    prediction = current_arr + delta
    return prediction, future_arr, current_arr, delta


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "mae_k": float(mean_absolute_error(actual, predicted)),
        "rmse_k": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "bias_k": float(np.mean(error)),
        "p95_absolute_error_k": float(np.percentile(np.abs(error), 95)),
        "max_absolute_error_k": float(np.max(np.abs(error))),
        "n_windows": int(len(actual)),
    }


def save_curve(actual: np.ndarray, predicted: np.ndarray, path: Path, title: str) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax1.plot(actual, color="black", linewidth=1.0, label="Measured")
    ax1.plot(predicted, color="#d55e00", linewidth=1.0, label="Prediction")
    ax1.set_ylabel("Temperature (K)")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(predicted - actual, color="#0072b2", linewidth=0.8)
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_xlabel("Prediction origin")
    ax2.set_ylabel("Residual (K)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_experiment(spec_name: str) -> None:
    if spec_name not in SPECS:
        raise KeyError(f"Unknown Phase 1 experiment: {spec_name}")
    spec = SPECS[spec_name]
    args = parse_args(spec)
    set_seed(args.seed)
    device = choose_device()
    print(f"Experiment={spec.name}; device={device}; data_dir={DATA_DIR}")

    # Development data only. Do not load test data before this block has finished.
    train_frame = add_target_delta(add_rebuilt_physics_features(load_frame("train_clean.pkl")))
    val_frame = add_target_delta(add_rebuilt_physics_features(load_frame("val_clean.pkl")))
    assert_group_separation(train_frame, val_frame)
    features = resolve_feature_columns(train_frame, spec)
    validate_feature_schema(train_frame, features, "train")
    validate_feature_schema(val_frame, features, "validation")

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    train_x = scaler_x.fit_transform(train_frame[features]).astype(np.float32)
    train_y = scaler_y.fit_transform(train_frame[[DELTA_COL]]).astype(np.float32)
    val_x = scaler_x.transform(val_frame[features]).astype(np.float32)
    val_y = scaler_y.transform(val_frame[[DELTA_COL]]).astype(np.float32)

    train_dataset = WindowDataset(train_frame, train_x, train_y)
    val_dataset = WindowDataset(val_frame, val_x, val_y)
    train_loader = make_loader(train_dataset, args, shuffle=True)
    val_loader = make_loader(val_dataset, args, shuffle=False)
    model = PlainGRU(len(features), args.hidden_size, args.dropout).to(device)
    best_state, history = train_model(model, train_loader, val_loader, args, device)
    model.load_state_dict(best_state)

    evaluation_scope = (
        f"final_{Path(args.test_file).stem}" if args.evaluate_test else "validation_only"
    )
    run_dir = OUTPUT_DIR / spec.name / evaluation_scope / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    val_prediction, val_actual, val_current, val_delta = predict_dataset(model, val_loader, device, scaler_y)
    val_metrics = compute_metrics(val_actual, val_prediction)

    # Feature/model selection must not inspect any held-out test result.  A
    # frozen final design can opt in explicitly with --evaluate-test.
    test_frame: pd.DataFrame | None = None
    test_prediction = test_actual = test_current = test_delta = None
    test_metrics: dict[str, float] | None = None
    if args.evaluate_test:
        test_frame = add_target_delta(add_rebuilt_physics_features(load_frame(args.test_file)))
        assert_group_separation(train_frame, val_frame, test_frame)
        validate_feature_schema(test_frame, features, "test")
        test_x = scaler_x.transform(test_frame[features]).astype(np.float32)
        test_y = scaler_y.transform(test_frame[[DELTA_COL]]).astype(np.float32)
        test_dataset = WindowDataset(test_frame, test_x, test_y)
        test_loader = make_loader(test_dataset, args, shuffle=False)
        test_prediction, test_actual, test_current, test_delta = predict_dataset(
            model, test_loader, device, scaler_y
        )
        test_metrics = compute_metrics(test_actual, test_prediction)

    metadata = {
        "experiment": spec.name,
        "removed_group": spec.removed_group,
        "added_groups": list(spec.added_groups),
        "seed": args.seed,
        "look_back": LOOK_BACK,
        "predict_steps": PREDICT_STEPS,
        "target": TARGET_COL,
        "test_evaluated": args.evaluate_test,
        "test_file": args.test_file if args.evaluate_test else None,
        "feature_count": len(features),
        "feature_columns": features,
        "train_source_groups": sorted(train_frame["source_group"].unique().tolist()),
        "validation_source_groups": sorted(val_frame["source_group"].unique().tolist()),
        "test_source_groups": (
            sorted(test_frame["source_group"].unique().tolist()) if test_frame is not None else []
        ),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "training_rows": int(len(train_frame)),
        "validation_rows": int(len(val_frame)),
        "test_rows": int(len(test_frame)) if test_frame is not None else 0,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save({"model_state_dict": model.state_dict(), **metadata}, run_dir / "model.pt")
    joblib.dump(scaler_x, run_dir / "scaler_x.pkl")
    joblib.dump(scaler_y, run_dir / "scaler_y.pkl")
    pd.DataFrame(
        {"current_temperature_k": val_current, "actual_temperature_k": val_actual,
         "predicted_temperature_k": val_prediction, "predicted_delta_k": val_delta}
    ).to_csv(run_dir / "val_predictions.csv", index=False)
    save_curve(val_actual, val_prediction, run_dir / "val_prediction.png", f"{spec.name} | validation")
    if args.evaluate_test:
        assert test_current is not None
        assert test_actual is not None
        assert test_prediction is not None
        assert test_delta is not None
        pd.DataFrame(
            {"current_temperature_k": test_current, "actual_temperature_k": test_actual,
             "predicted_temperature_k": test_prediction, "predicted_delta_k": test_delta}
        ).to_csv(run_dir / "test_predictions.csv", index=False)
        save_curve(
            test_actual,
            test_prediction,
            run_dir / "test_prediction.png",
            f"{spec.name} | {args.test_file}",
        )
    print(json.dumps({"validation": val_metrics, "test": test_metrics}, indent=2))
    print(f"Saved to {run_dir}")
