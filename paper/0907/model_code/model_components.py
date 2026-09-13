#!/usr/bin/env python3
"""Pure-PyTorch model components for the Thv model ablation study.

The graph is a fixed directed acyclic representation of the documented pipe
logic.  Valve commands live on edges; temperatures, pressures and flow live on
nodes.  No data-dependent KNN graph is constructed.  This keeps the model
compatible with CPU, CUDA and Apple MPS without torch-geometric or a third-party
KAN package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


NODE_NAMES = (
    "G1_source_8300",
    "G2_source_8313",
    "G_distributor",
    "G_main_8310",
    "screen_branch_8311",
    "screen_branch_8312",
    "A_source_8351",
    "postmix_TE8351",
    "mainline_TE8352",
    "module_inlet_TE8353",
    "module_average_Thv",
)

# (source node, destination node, controlling valve or None).  The order is
# topological and is used by DirectedDAGSweep to propagate an upstream change
# through the complete known path in one sweep.
EDGE_SPECS = (
    (0, 2, "CV8300"),
    (1, 2, "CV8313"),
    (2, 3, "CV8310"),
    (2, 4, "CV8311"),
    (2, 5, "CV8312"),
    (3, 7, None),
    (6, 7, "CV8351"),
    (7, 8, None),
    (8, 9, None),
    (9, 10, "EC-V2"),
    (9, 10, "COOLDOWN"),
)

# Incoming edges grouped by target, also in topological order.
TARGET_EDGE_GROUPS = (
    (2, (0, 1)),
    (3, (2,)),
    (4, (3,)),
    (5, (4,)),
    (7, (5, 6)),
    (8, (7,)),
    (9, (8,)),
    (10, (9, 10)),
)

MODEL_NAMES = (
    "gru_baseline",
    "lstm_baseline",
    "serial_mlp_gnn_gru",
    "serial_kan_gnn_gru",
    "parallel_gru_mlp_gnn",
    "parallel_gru_kan_gnn",
    "parallel_lstm_mlp_gnn",
    "parallel_lstm_kan_gnn",
    "tcn_baseline",
    "serial_mlp_gnn_tcn",
    "serial_kan_gnn_tcn",
)


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    history_hidden: int = 64
    graph_hidden: int = 64
    graph_sweeps: int = 1
    edge_hidden: int = 32
    kan_grid: int = 8
    tcn_levels: int = 3
    tcn_kernel: int = 3
    control_hidden: int = 32
    fusion_hidden: int = 64
    dropout: float = 0.1

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "model_name": self.model_name,
            "history_hidden": self.history_hidden,
            "graph_hidden": self.graph_hidden,
            "graph_sweeps": self.graph_sweeps,
            "edge_hidden": self.edge_hidden,
            "kan_grid": self.kan_grid,
            "tcn_levels": self.tcn_levels,
            "tcn_kernel": self.tcn_kernel,
            "control_hidden": self.control_hidden,
            "fusion_hidden": self.fusion_hidden,
            "dropout": self.dropout,
        }


class ControlEncoder(nn.Module):
    def __init__(
        self,
        control_horizon: int,
        control_input_size: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.control_horizon = int(control_horizon)
        self.control_input_size = int(control_input_size)
        self.network = nn.Sequential(
            nn.Linear(self.control_horizon * self.control_input_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, controls: torch.Tensor) -> torch.Tensor:
        if controls.shape[1] != self.control_horizon:
            raise ValueError(
                f"Expected {self.control_horizon} future control rows, "
                f"received {controls.shape[1]}"
            )
        return self.network(controls.flatten(start_dim=1))


class MonotonicSplineKAN(nn.Module):
    """One-dimensional monotonic B-spline/KAN edge functions.

    Each graph edge owns a piecewise-linear spline over valve opening [0, 1].
    Positive knot increments guarantee a non-decreasing response to the edge's
    own opening.  Context variables are deliberately handled by a separate
    unconstrained MLP because pressure, temperature and sibling-valve effects
    should not all be forced to be monotonic.
    """

    def __init__(self, edge_count: int, grid_size: int) -> None:
        super().__init__()
        if grid_size < 3:
            raise ValueError("KAN grid_size must be at least 3")
        self.edge_count = int(edge_count)
        self.grid_size = int(grid_size)
        self.raw_increments = nn.Parameter(
            torch.zeros(self.edge_count, self.grid_size - 1)
        )
        self.raw_scale = nn.Parameter(torch.zeros(self.edge_count))
        self.raw_leak = nn.Parameter(torch.full((self.edge_count,), -6.0))

    def forward(self, opening: torch.Tensor) -> torch.Tensor:
        if opening.ndim != 2 or opening.shape[1] != self.edge_count:
            raise ValueError("opening must have shape [batch, edge_count]")
        positive = F.softplus(self.raw_increments) + 1e-6
        cumulative = torch.cumsum(positive, dim=1)
        normalized = cumulative / cumulative[:, -1:].clamp_min(1e-6)
        zero = torch.zeros(
            self.edge_count,
            1,
            device=opening.device,
            dtype=opening.dtype,
        )
        knots = torch.cat((zero, normalized.to(opening.dtype)), dim=1)

        position = opening.clamp(0.0, 1.0) * float(self.grid_size - 1)
        lower = torch.floor(position).to(torch.long).clamp(0, self.grid_size - 2)
        fraction = position - lower.to(position.dtype)
        edge_ids = torch.arange(self.edge_count, device=opening.device).view(1, -1)
        lower_value = knots[edge_ids, lower]
        upper_value = knots[edge_ids, lower + 1]
        interpolated = lower_value + fraction * (upper_value - lower_value)

        scale = F.softplus(self.raw_scale).view(1, -1) + 1e-4
        leak = 0.01 * F.softplus(self.raw_leak).view(1, -1)
        return leak + scale * interpolated


class EdgeGate(nn.Module):
    def __init__(
        self,
        mode: str,
        edge_count: int,
        edge_attr_size: int,
        edge_hidden: int,
        kan_grid: int,
    ) -> None:
        super().__init__()
        if mode not in {"mlp", "kan"}:
            raise ValueError(f"Unknown edge gate mode: {mode}")
        self.mode = mode
        self.edge_count = int(edge_count)
        self.edge_embedding = nn.Embedding(edge_count, 8)
        if mode == "mlp":
            self.context_network = nn.Sequential(
                nn.Linear(edge_attr_size + 8, edge_hidden),
                nn.SiLU(),
                nn.Linear(edge_hidden, 1),
            )
            self.valve_kan = None
        else:
            # edge_attr[:, :, 0] is the edge's own valve opening.  Everything
            # else remains in an unconstrained context network.
            self.context_network = nn.Sequential(
                nn.Linear(edge_attr_size - 1 + 8, edge_hidden),
                nn.SiLU(),
                nn.Linear(edge_hidden, 1),
            )
            self.valve_kan = MonotonicSplineKAN(edge_count, kan_grid)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        batch = edge_attr.shape[0]
        edge_ids = torch.arange(self.edge_count, device=edge_attr.device)
        embedding = self.edge_embedding(edge_ids).unsqueeze(0).expand(batch, -1, -1)
        if self.mode == "mlp":
            logits = self.context_network(torch.cat((edge_attr, embedding), dim=-1))
            return F.softplus(logits.squeeze(-1)) + 1e-4
        assert self.valve_kan is not None
        valve_gate = self.valve_kan(edge_attr[:, :, 0])
        context = self.context_network(
            torch.cat((edge_attr[:, :, 1:], embedding), dim=-1)
        )
        return valve_gate * (F.softplus(context.squeeze(-1)) + 1e-4)


class DirectedDAGSweep(nn.Module):
    """One topological message-passing sweep over the complete directed DAG."""

    def __init__(
        self,
        hidden_size: int,
        gate_mode: str,
        edge_attr_size: int,
        edge_hidden: int,
        kan_grid: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.edge_gate = EdgeGate(
            gate_mode,
            len(EDGE_SPECS),
            edge_attr_size,
            edge_hidden,
            kan_grid,
        )
        self.message = nn.Linear(hidden_size, hidden_size, bias=False)
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.normalization = nn.ModuleList(
            nn.LayerNorm(hidden_size) for _ in NODE_NAMES
        )

    def forward(self, nodes: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        gates = self.edge_gate(edge_attr)
        states = [nodes[:, index, :] for index in range(len(NODE_NAMES))]
        for target, edge_indices in TARGET_EDGE_GROUPS:
            messages = []
            for edge_index in edge_indices:
                source = EDGE_SPECS[edge_index][0]
                message = self.message(states[source])
                messages.append(message * gates[:, edge_index].unsqueeze(-1))
            aggregate = torch.stack(messages, dim=0).sum(dim=0)
            update = self.update(torch.cat((states[target], aggregate), dim=-1))
            states[target] = self.normalization[target](states[target] + update)
        return torch.stack(states, dim=1)


class PhysicalGraphEncoder(nn.Module):
    """Build fixed node/edge tensors from one standardized process snapshot."""

    LOCAL_CHANNELS = 10
    CONTEXT_FEATURES = (
        "PT8310",
        "PT8351",
        "PT8352",
        "FT8351",
        "TE8310",
        "TE8351",
        "TE8352",
        "TE8353",
        "Thv",
    )
    SIBLING_VALVES = ("CV8310", "CV8311", "CV8312", "EC-V2", "COOLDOWN")

    def __init__(
        self,
        feature_names: Sequence[str],
        history_mean: np.ndarray,
        history_scale: np.ndarray,
        hidden_size: int,
        sweeps: int,
        gate_mode: str,
        edge_hidden: int,
        kan_grid: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if sweeps < 1:
            raise ValueError("graph sweeps must be at least 1")
        self.feature_names = tuple(feature_names)
        self.feature_index = {name: index for index, name in enumerate(feature_names)}
        required = {
            "TE8310", "TE8351", "TE8352", "TE8353", "Thv", "Tef", "Tcd",
            "DTbr", "FT8351", "PT8310", "PT8351", "PT8352", "CV8300",
            "CV8313", "CV8310", "CV8311", "CV8312", "CV8351", "EC-V2",
            "COOLDOWN",
        }
        missing = required - set(self.feature_names)
        if missing:
            raise KeyError(f"Graph encoder is missing input features: {sorted(missing)}")
        mean = np.asarray(history_mean, dtype=np.float32).reshape(-1)
        scale = np.asarray(history_scale, dtype=np.float32).reshape(-1)
        if len(mean) != len(feature_names) or len(scale) != len(feature_names):
            raise ValueError("History scaler size does not match feature_names")
        self.register_buffer("history_mean", torch.from_numpy(mean))
        self.register_buffer("history_scale", torch.from_numpy(scale))

        self.node_encoder = nn.Sequential(
            nn.Linear(2 * self.LOCAL_CHANNELS, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
        )
        self.node_embedding = nn.Embedding(len(NODE_NAMES), hidden_size)
        edge_attr_size = 2 + len(self.CONTEXT_FEATURES) + len(self.SIBLING_VALVES)
        self.sweeps = nn.ModuleList(
            DirectedDAGSweep(
                hidden_size,
                gate_mode,
                edge_attr_size,
                edge_hidden,
                kan_grid,
                dropout,
            )
            for _ in range(sweeps)
        )
        self.readout = nn.Sequential(
            nn.Linear(4 * hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def _scaled(self, snapshot: torch.Tensor, name: str) -> torch.Tensor:
        return snapshot[:, self.feature_index[name]]

    def _raw(self, snapshot: torch.Tensor, name: str) -> torch.Tensor:
        index = self.feature_index[name]
        return (
            snapshot[:, index] * self.history_scale[index] + self.history_mean[index]
        )

    def _opening(self, snapshot: torch.Tensor, name: str | None) -> torch.Tensor:
        if name is None:
            return torch.ones(snapshot.shape[0], device=snapshot.device, dtype=snapshot.dtype)
        return (self._raw(snapshot, name) / 100.0).clamp(0.0, 1.0)

    def _build_nodes(self, snapshot: torch.Tensor) -> torch.Tensor:
        batch = snapshot.shape[0]
        values = torch.zeros(
            batch,
            len(NODE_NAMES),
            self.LOCAL_CHANNELS,
            device=snapshot.device,
            dtype=snapshot.dtype,
        )
        masks = torch.zeros_like(values)

        assignments: Mapping[int, tuple[tuple[int, str], ...]] = {
            0: ((0, "TE8310"), (1, "PT8310"), (2, "FT8351"), (5, "CV8300")),
            1: ((0, "TE8310"), (1, "PT8310"), (2, "FT8351"), (5, "CV8313")),
            2: ((0, "TE8310"), (1, "PT8310"), (2, "FT8351"), (5, "CV8310"), (6, "CV8311"), (7, "CV8312")),
            3: ((0, "TE8351"), (1, "PT8351"), (2, "FT8351"), (3, "TE8310"), (5, "CV8310")),
            4: ((0, "TE8310"), (1, "PT8310"), (5, "CV8311")),
            5: ((0, "TE8310"), (1, "PT8310"), (5, "CV8312")),
            6: ((0, "TE8351"), (1, "PT8351"), (5, "CV8351")),
            7: ((0, "TE8351"), (1, "PT8351"), (2, "FT8351"), (3, "TE8310"), (5, "CV8351")),
            8: ((0, "TE8352"), (1, "PT8352"), (2, "FT8351"), (3, "TE8351")),
            9: ((0, "TE8353"), (1, "PT8352"), (2, "FT8351"), (3, "TE8352")),
            10: ((0, "Thv"), (2, "FT8351"), (3, "TE8353"), (4, "TE8310"), (5, "EC-V2"), (6, "COOLDOWN"), (7, "DTbr"), (8, "Tef"), (9, "Tcd")),
        }
        for node, node_assignments in assignments.items():
            for channel, feature in node_assignments:
                values[:, node, channel] = self._scaled(snapshot, feature)
                masks[:, node, channel] = 1.0
        encoded = self.node_encoder(torch.cat((values, masks), dim=-1))
        node_ids = torch.arange(len(NODE_NAMES), device=snapshot.device)
        return encoded + self.node_embedding(node_ids).unsqueeze(0)

    def _build_edges(
        self,
        snapshot: torch.Tensor,
        previous_snapshot: torch.Tensor,
    ) -> torch.Tensor:
        openings = torch.stack(
            [self._opening(snapshot, valve) for _, _, valve in EDGE_SPECS], dim=1
        )
        previous_openings = torch.stack(
            [self._opening(previous_snapshot, valve) for _, _, valve in EDGE_SPECS],
            dim=1,
        )
        opening_change = openings - previous_openings
        context = torch.stack(
            [self._scaled(snapshot, name) for name in self.CONTEXT_FEATURES], dim=1
        )
        context = context.unsqueeze(1).expand(-1, len(EDGE_SPECS), -1)
        siblings = torch.stack(
            [self._opening(snapshot, name) for name in self.SIBLING_VALVES], dim=1
        )
        siblings = siblings.unsqueeze(1).expand(-1, len(EDGE_SPECS), -1)
        return torch.cat(
            (openings.unsqueeze(-1), opening_change.unsqueeze(-1), context, siblings),
            dim=-1,
        )

    def forward(
        self,
        snapshot: torch.Tensor,
        previous_snapshot: torch.Tensor,
    ) -> torch.Tensor:
        nodes = self._build_nodes(snapshot)
        edge_attr = self._build_edges(snapshot, previous_snapshot)
        for sweep in self.sweeps:
            nodes = sweep(nodes, edge_attr)
        target = nodes[:, 10, :]
        postmix = nodes[:, 7, :]
        inlet = nodes[:, 9, :]
        pooled = nodes.mean(dim=1)
        return self.readout(torch.cat((target, postmix, inlet, pooled), dim=-1))


class CausalTemporalBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.convolution = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.normalization = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        convolved = self.convolution(F.pad(sequence, (self.left_padding, 0)))
        activated = self.dropout(F.silu(convolved))
        residual = sequence + activated
        return self.normalization(residual.transpose(1, 2)).transpose(1, 2)


class CausalTCNEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        levels: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError("TCN levels must be at least 1")
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.blocks = nn.ModuleList(
            CausalTemporalBlock(hidden_size, kernel_size, 2**level, dropout)
            for level in range(levels)
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        sequence = self.input_projection(history).transpose(1, 2)
        for block in self.blocks:
            sequence = block(sequence)
        return sequence[:, :, -1]


class ResidualGRUBaseline(nn.Module):
    def __init__(self, history_input_size: int, control_input_size: int, control_horizon: int, config: ModelConfig) -> None:
        super().__init__()
        self.history = nn.GRU(history_input_size, config.history_hidden, batch_first=True)
        self.control = ControlEncoder(control_horizon, control_input_size, config.control_hidden, config.dropout)
        self.fusion = nn.Sequential(
            nn.Linear(config.history_hidden + config.control_hidden, config.fusion_hidden),
            nn.SiLU(), nn.Dropout(config.dropout), nn.Linear(config.fusion_hidden, 1),
        )

    def forward(self, history: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        output, _ = self.history(history)
        state = output[:, -1, :]
        return self.fusion(torch.cat((state, self.control(controls)), dim=-1)).squeeze(-1)


class ResidualLSTMBaseline(nn.Module):
    """LSTM temporal baseline with the same control and fusion heads as GRU."""

    def __init__(self, history_input_size: int, control_input_size: int, control_horizon: int, config: ModelConfig) -> None:
        super().__init__()
        self.history = nn.LSTM(history_input_size, config.history_hidden, batch_first=True)
        self.control = ControlEncoder(control_horizon, control_input_size, config.control_hidden, config.dropout)
        self.fusion = nn.Sequential(
            nn.Linear(config.history_hidden + config.control_hidden, config.fusion_hidden),
            nn.SiLU(), nn.Dropout(config.dropout), nn.Linear(config.fusion_hidden, 1),
        )

    def forward(self, history: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        output, _ = self.history(history)
        state = output[:, -1, :]
        return self.fusion(torch.cat((state, self.control(controls)), dim=-1)).squeeze(-1)


class ResidualTCNBaseline(nn.Module):
    def __init__(self, history_input_size: int, control_input_size: int, control_horizon: int, config: ModelConfig) -> None:
        super().__init__()
        self.history = CausalTCNEncoder(history_input_size, config.history_hidden, config.tcn_levels, config.tcn_kernel, config.dropout)
        self.control = ControlEncoder(control_horizon, control_input_size, config.control_hidden, config.dropout)
        self.fusion = nn.Sequential(
            nn.Linear(config.history_hidden + config.control_hidden, config.fusion_hidden),
            nn.SiLU(), nn.Dropout(config.dropout), nn.Linear(config.fusion_hidden, 1),
        )

    def forward(self, history: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat((self.history(history), self.control(controls)), dim=-1)).squeeze(-1)


class SerialGraphTemporal(nn.Module):
    def __init__(
        self,
        temporal_type: str,
        gate_mode: str,
        feature_names: Sequence[str],
        history_mean: np.ndarray,
        history_scale: np.ndarray,
        control_input_size: int,
        control_horizon: int,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.graph = PhysicalGraphEncoder(
            feature_names, history_mean, history_scale, config.graph_hidden,
            config.graph_sweeps, gate_mode, config.edge_hidden, config.kan_grid,
            config.dropout,
        )
        self.temporal_type = temporal_type
        if temporal_type == "gru":
            self.temporal = nn.GRU(config.graph_hidden, config.history_hidden, batch_first=True)
        elif temporal_type == "tcn":
            self.temporal = CausalTCNEncoder(
                config.graph_hidden, config.history_hidden, config.tcn_levels,
                config.tcn_kernel, config.dropout,
            )
        else:
            raise ValueError(f"Unknown temporal type: {temporal_type}")
        self.control = ControlEncoder(control_horizon, control_input_size, config.control_hidden, config.dropout)
        self.fusion = nn.Sequential(
            nn.Linear(config.history_hidden + config.control_hidden, config.fusion_hidden),
            nn.SiLU(), nn.Dropout(config.dropout), nn.Linear(config.fusion_hidden, 1),
        )

    def forward(self, history: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        batch, length, features = history.shape
        previous = torch.cat((history[:, :1, :], history[:, :-1, :]), dim=1)
        graph_sequence = self.graph(
            history.reshape(batch * length, features),
            previous.reshape(batch * length, features),
        ).reshape(batch, length, -1)
        if self.temporal_type == "gru":
            output, _ = self.temporal(graph_sequence)
            temporal_state = output[:, -1, :]
        else:
            temporal_state = self.temporal(graph_sequence)
        return self.fusion(
            torch.cat((temporal_state, self.control(controls)), dim=-1)
        ).squeeze(-1)


class ParallelRNNGraph(nn.Module):
    def __init__(
        self,
        temporal_type: str,
        gate_mode: str,
        feature_names: Sequence[str],
        history_mean: np.ndarray,
        history_scale: np.ndarray,
        control_input_size: int,
        control_horizon: int,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.temporal_type = temporal_type
        if temporal_type == "gru":
            self.history = nn.GRU(
                len(feature_names), config.history_hidden, batch_first=True
            )
        elif temporal_type == "lstm":
            self.history = nn.LSTM(
                len(feature_names), config.history_hidden, batch_first=True
            )
        else:
            raise ValueError(f"Unknown temporal type: {temporal_type}")
        self.graph = PhysicalGraphEncoder(
            feature_names, history_mean, history_scale, config.graph_hidden,
            config.graph_sweeps, gate_mode, config.edge_hidden, config.kan_grid,
            config.dropout,
        )
        self.control = ControlEncoder(control_horizon, control_input_size, config.control_hidden, config.dropout)
        fusion_input = config.history_hidden + config.graph_hidden + config.control_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, config.fusion_hidden),
            nn.SiLU(), nn.Dropout(config.dropout), nn.Linear(config.fusion_hidden, 1),
        )

    def forward(self, history: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        history_output, _ = self.history(history)
        history_state = history_output[:, -1, :]
        previous = history[:, -2, :] if history.shape[1] > 1 else history[:, -1, :]
        graph_state = self.graph(history[:, -1, :], previous)
        control_state = self.control(controls)
        return self.fusion(
            torch.cat((history_state, graph_state, control_state), dim=-1)
        ).squeeze(-1)


def build_model(
    config: ModelConfig,
    feature_names: Sequence[str],
    control_names: Sequence[str],
    control_horizon: int,
    history_mean: np.ndarray,
    history_scale: np.ndarray,
) -> nn.Module:
    if config.model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model_name={config.model_name}")
    if config.model_name == "gru_baseline":
        return ResidualGRUBaseline(len(feature_names), len(control_names), control_horizon, config)
    if config.model_name == "lstm_baseline":
        return ResidualLSTMBaseline(len(feature_names), len(control_names), control_horizon, config)
    if config.model_name == "tcn_baseline":
        return ResidualTCNBaseline(len(feature_names), len(control_names), control_horizon, config)
    if config.model_name == "serial_mlp_gnn_gru":
        return SerialGraphTemporal("gru", "mlp", feature_names, history_mean, history_scale, len(control_names), control_horizon, config)
    if config.model_name == "serial_kan_gnn_gru":
        return SerialGraphTemporal("gru", "kan", feature_names, history_mean, history_scale, len(control_names), control_horizon, config)
    if config.model_name == "serial_kan_gnn_tcn":
        return SerialGraphTemporal("tcn", "kan", feature_names, history_mean, history_scale, len(control_names), control_horizon, config)
    if config.model_name == "serial_mlp_gnn_tcn":
        return SerialGraphTemporal("tcn", "mlp", feature_names, history_mean, history_scale, len(control_names), control_horizon, config)
    if config.model_name == "parallel_gru_mlp_gnn":
        return ParallelRNNGraph("gru", "mlp", feature_names, history_mean, history_scale, len(control_names), control_horizon, config)
    if config.model_name == "parallel_gru_kan_gnn":
        return ParallelRNNGraph("gru", "kan", feature_names, history_mean, history_scale, len(control_names), control_horizon, config)
    if config.model_name == "parallel_lstm_mlp_gnn":
        return ParallelRNNGraph("lstm", "mlp", feature_names, history_mean, history_scale, len(control_names), control_horizon, config)
    if config.model_name == "parallel_lstm_kan_gnn":
        return ParallelRNNGraph("lstm", "kan", feature_names, history_mean, history_scale, len(control_names), control_horizon, config)
    raise AssertionError("Unhandled model name")


def model_description(model_name: str) -> str:
    descriptions = {
        "gru_baseline": "Flat historical GRU plus known-future-control encoder",
        "lstm_baseline": "Flat historical LSTM plus known-future-control encoder",
        "serial_mlp_gnn_gru": "Directed edge-MLP DAG snapshots, then GRU",
        "serial_kan_gnn_gru": "Monotonic valve-KAN directed DAG snapshots, then GRU",
        "parallel_gru_mlp_gnn": "Flat GRU parallel with current-snapshot edge-MLP DAG",
        "parallel_gru_kan_gnn": "Flat GRU parallel with current-snapshot valve-KAN DAG",
        "parallel_lstm_mlp_gnn": "Flat LSTM parallel with current-snapshot edge-MLP DAG",
        "parallel_lstm_kan_gnn": "Flat LSTM parallel with current-snapshot valve-KAN DAG",
        "tcn_baseline": "Causal dilated TCN plus known-future-control encoder",
        "serial_mlp_gnn_tcn": "Directed edge-MLP DAG snapshots, then causal TCN",
        "serial_kan_gnn_tcn": "Monotonic valve-KAN directed DAG snapshots, then causal TCN",
    }
    return descriptions[model_name]
