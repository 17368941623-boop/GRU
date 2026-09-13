#!/usr/bin/env python3
"""Parallel GRU + lag-aware GNN with monotonic KAN valve gates."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from protocol import FUTURE_CONTROL_COLUMNS, HISTORY_COLUMNS, RAW_COLUMNS, VALVE_COLUMNS


EDGE_TYPE_ID = {"physical": 0, "data": 1, "both": 2, "random": 3}
CONFIDENCE_VALUE = {"low": 0.25, "medium": 0.55, "high": 0.85, "forced": 0.40}


class MonotonicEdgeSpline(nn.Module):
    """Independent non-decreasing piecewise-linear functions for graph edges."""

    def __init__(self, edge_count: int, grid_size: int = 8) -> None:
        super().__init__()
        if grid_size < 3:
            raise ValueError("grid_size must be at least 3")
        self.edge_count = int(edge_count)
        self.grid_size = int(grid_size)
        self.raw_increments = nn.Parameter(torch.zeros(edge_count, grid_size - 1))
        self.raw_scale = nn.Parameter(torch.zeros(edge_count))
        self.raw_floor = nn.Parameter(torch.full((edge_count,), -6.0))

    def forward(self, opening_01: torch.Tensor) -> torch.Tensor:
        if opening_01.ndim < 2 or opening_01.shape[-1] != self.edge_count:
            raise ValueError("opening_01 must end with edge_count")
        increments = F.softplus(self.raw_increments) + 1e-6
        knots = torch.cumsum(increments, dim=-1)
        knots = knots / knots[:, -1:].clamp_min(1e-8)
        knots = torch.cat((torch.zeros_like(knots[:, :1]), knots), dim=-1)
        position = opening_01.clamp(0.0, 1.0) * (self.grid_size - 1)
        lower = position.floor().long().clamp(0, self.grid_size - 2)
        fraction = position - lower.to(position.dtype)
        edge_ids = torch.arange(self.edge_count, device=opening_01.device).view(
            *((1,) * (opening_01.ndim - 1)), self.edge_count
        )
        lo = knots[edge_ids, lower]
        hi = knots[edge_ids, lower + 1]
        value = lo + fraction * (hi - lo)
        parameter_shape = (1,) * (opening_01.ndim - 1) + (self.edge_count,)
        floor = F.softplus(self.raw_floor).view(parameter_shape)
        scale = F.softplus(self.raw_scale).view(parameter_shape)
        return 0.01 * floor + scale * value


class LagGraphEncoder(nn.Module):
    """Message passing over variable pairs whose edges carry explicit delays."""

    def __init__(
        self,
        edges: list[dict[str, Any]],
        history_mean: np.ndarray,
        history_scale: np.ndarray,
        gate_mode: str = "kan",
        hidden_size: int = 64,
        max_lag: int = 30,
        grid_size: int = 8,
        sweeps: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if gate_mode not in {"kan", "mlp"}:
            raise ValueError("gate_mode must be kan or mlp")
        if not edges:
            raise ValueError("LagGraphEncoder requires at least one edge")
        self.gate_mode = gate_mode
        self.hidden_size = int(hidden_size)
        self.max_lag = int(max_lag)
        self.sweeps = int(sweeps)
        names = {name: index for index, name in enumerate(RAW_COLUMNS)}
        source = [names[str(edge["source"])] for edge in edges]
        destination = [names[str(edge["destination"])] for edge in edges]
        lag = [int(edge["lag_steps"]) for edge in edges]
        if min(lag) < 1 or max(lag) > max_lag:
            raise ValueError(f"Every graph lag must be within [1, {max_lag}]")
        confidence = [
            float(edge.get("confidence_score", CONFIDENCE_VALUE.get(str(edge.get("confidence", "medium")), 0.5)))
            for edge in edges
        ]
        edge_type = [EDGE_TYPE_ID.get(str(edge.get("edge_type", "data")), 1) for edge in edges]
        valve = [str(edge["source"]) in VALVE_COLUMNS for edge in edges]
        self.register_buffer("edge_source", torch.tensor(source, dtype=torch.long))
        self.register_buffer("edge_destination", torch.tensor(destination, dtype=torch.long))
        self.register_buffer("edge_lag", torch.tensor(lag, dtype=torch.long))
        self.register_buffer("edge_confidence", torch.tensor(confidence, dtype=torch.float32))
        self.register_buffer("edge_type", torch.tensor(edge_type, dtype=torch.long))
        self.register_buffer("valve_edge", torch.tensor(valve, dtype=torch.bool))
        self.register_buffer("raw_mean", torch.tensor(history_mean[: len(RAW_COLUMNS)], dtype=torch.float32))
        self.register_buffer("raw_scale", torch.tensor(history_scale[: len(RAW_COLUMNS)], dtype=torch.float32))

        node_emb = 12
        edge_emb = 6
        self.node_embedding = nn.Embedding(len(RAW_COLUMNS), node_emb)
        self.edge_type_embedding = nn.Embedding(len(EDGE_TYPE_ID), edge_emb)
        self.node_encoder = nn.Sequential(nn.Linear(1 + node_emb, hidden_size), nn.SiLU())
        message_input = hidden_size * 2 + 1 + 1 + 1 + edge_emb
        self.message_network = nn.Sequential(
            nn.Linear(message_input, hidden_size), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.update_network = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.pool_score = nn.Linear(hidden_size, 1)
        if gate_mode == "kan":
            self.edge_gate = MonotonicEdgeSpline(len(edges), grid_size)
        else:
            self.edge_gate = nn.Sequential(
                nn.Linear(1 + 1 + 1 + edge_emb, 24), nn.SiLU(), nn.Linear(24, 1)
            )

    def _lagged_source(self, raw_history: torch.Tensor) -> torch.Tensor:
        batch, length, _ = raw_history.shape
        time_index = length - 1 - self.edge_lag
        if int(time_index.min()) < 0:
            raise ValueError("Graph lag exceeds supplied history length")
        batch_index = torch.arange(batch, device=raw_history.device)[:, None]
        return raw_history[batch_index, time_index[None, :], self.edge_source[None, :]]

    def _gates(self, lagged_source: torch.Tensor, type_embedding: torch.Tensor) -> torch.Tensor:
        lag_norm = self.edge_lag.float()[None, :] / float(self.max_lag)
        confidence = self.edge_confidence[None, :]
        physical = lagged_source * self.raw_scale[self.edge_source][None, :] + self.raw_mean[self.edge_source][None, :]
        opening = (physical / 100.0).clamp(0.0, 1.0)
        if self.gate_mode == "kan":
            learned = self.edge_gate(opening)
            return torch.where(self.valve_edge[None, :], learned, torch.ones_like(learned))
        gate_features = torch.cat(
            (
                lagged_source.unsqueeze(-1),
                lag_norm.expand_as(lagged_source).unsqueeze(-1),
                confidence.expand_as(lagged_source).unsqueeze(-1),
                type_embedding.expand(lagged_source.shape[0], -1, -1),
            ),
            dim=-1,
        )
        return F.softplus(self.edge_gate(gate_features).squeeze(-1)) + 1e-4

    def forward(self, raw_history: torch.Tensor) -> torch.Tensor:
        current = raw_history[:, -1, :]
        batch = current.shape[0]
        node_ids = torch.arange(len(RAW_COLUMNS), device=current.device)
        embedding = self.node_embedding(node_ids)[None, :, :].expand(batch, -1, -1)
        state = self.node_encoder(torch.cat((current.unsqueeze(-1), embedding), dim=-1))
        lagged_source = self._lagged_source(raw_history)
        type_embedding = self.edge_type_embedding(self.edge_type)[None, :, :]
        lag_norm = self.edge_lag.float()[None, :, None] / float(self.max_lag)
        confidence = self.edge_confidence[None, :, None]
        gate = self._gates(lagged_source, type_embedding)
        destination_index = self.edge_destination[None, :, None].expand(batch, -1, self.hidden_size)
        for _ in range(self.sweeps):
            source_state = state[:, self.edge_source, :]
            destination_state = state[:, self.edge_destination, :]
            message_features = torch.cat(
                (
                    source_state,
                    destination_state,
                    lagged_source.unsqueeze(-1),
                    lag_norm.expand(batch, -1, -1),
                    confidence.expand(batch, -1, -1),
                    type_embedding.expand(batch, -1, -1),
                ),
                dim=-1,
            )
            messages = self.message_network(message_features) * gate.unsqueeze(-1)
            aggregate = torch.zeros_like(state)
            aggregate.scatter_add_(1, destination_index, messages)
            state = self.norm(state + self.update_network(torch.cat((state, aggregate), dim=-1)))
        attention = torch.softmax(self.pool_score(state).squeeze(-1), dim=1)
        return torch.sum(state * attention.unsqueeze(-1), dim=1)


class ParallelTrajectoryModel(nn.Module):
    """Direct trajectory model; kth output sees only controls u(t)..u(t+k-1)."""

    def __init__(
        self,
        horizon: int,
        history_mean: np.ndarray,
        history_scale: np.ndarray,
        control_mean: np.ndarray,
        control_scale: np.ndarray,
        edges: list[dict[str, Any]] | None,
        gate_mode: str,
        history_hidden: int = 64,
        graph_hidden: int = 64,
        control_hidden: int = 32,
        fusion_hidden: int = 96,
        horizon_embedding: int = 8,
        dropout: float = 0.10,
        max_lag: int = 30,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.graph_enabled = edges is not None
        self.gate_mode = gate_mode
        self.history_gru = nn.GRU(len(HISTORY_COLUMNS), history_hidden, batch_first=True)
        self.register_buffer("control_mean", torch.tensor(control_mean, dtype=torch.float32))
        self.register_buffer("control_scale", torch.tensor(control_scale, dtype=torch.float32))
        if gate_mode == "kan":
            self.control_mapping = MonotonicEdgeSpline(len(FUTURE_CONTROL_COLUMNS), grid_size=8)
            control_input = len(FUTURE_CONTROL_COLUMNS) * 2
        elif gate_mode == "mlp":
            self.control_mapping = nn.Sequential(
                nn.Linear(len(FUTURE_CONTROL_COLUMNS), 24), nn.SiLU(),
                nn.Linear(24, len(FUTURE_CONTROL_COLUMNS)),
            )
            control_input = len(FUTURE_CONTROL_COLUMNS) * 2
        elif gate_mode == "none":
            self.control_mapping = None
            control_input = len(FUTURE_CONTROL_COLUMNS)
        else:
            raise ValueError("gate_mode must be none, kan or mlp")
        self.control_gru = nn.GRU(control_input, control_hidden, batch_first=True)
        if self.graph_enabled:
            self.graph = LagGraphEncoder(
                edges or [], history_mean, history_scale, gate_mode=gate_mode,
                hidden_size=graph_hidden, max_lag=max_lag, dropout=dropout,
            )
            graph_output = graph_hidden
        else:
            self.graph = None
            graph_output = 0
        self.horizon_embedding = nn.Embedding(horizon, horizon_embedding)
        fusion_input = history_hidden + control_hidden + graph_output + horizon_embedding
        self.head = nn.Sequential(
            nn.Linear(fusion_input, fusion_hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2), nn.SiLU(),
            nn.Linear(fusion_hidden // 2, 1),
        )

    def forward(self, history: torch.Tensor, future_controls: torch.Tensor) -> torch.Tensor:
        if future_controls.shape[1] != self.horizon:
            raise ValueError(f"Expected {self.horizon} control steps")
        _, history_state = self.history_gru(history)
        history_vector = history_state[-1]
        if self.control_mapping is None:
            control_features = future_controls
        elif self.gate_mode == "kan":
            physical = future_controls * self.control_scale[None, None, :] + self.control_mean[None, None, :]
            conductance = self.control_mapping((physical / 100.0).clamp(0.0, 1.0))
            control_features = torch.cat((future_controls, conductance), dim=-1)
        else:
            mapped = self.control_mapping(future_controls)
            control_features = torch.cat((future_controls, mapped), dim=-1)
        control_prefix, _ = self.control_gru(control_features)
        parts = [history_vector[:, None, :].expand(-1, self.horizon, -1), control_prefix]
        if self.graph is not None:
            graph_vector = self.graph(history[:, :, : len(RAW_COLUMNS)])
            parts.append(graph_vector[:, None, :].expand(-1, self.horizon, -1))
        horizon_ids = torch.arange(self.horizon, device=history.device)
        parts.append(self.horizon_embedding(horizon_ids)[None, :, :].expand(history.shape[0], -1, -1))
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


def build_model(
    horizon: int,
    history_mean: np.ndarray,
    history_scale: np.ndarray,
    control_mean: np.ndarray,
    control_scale: np.ndarray,
    edges: list[dict[str, Any]] | None,
    gate_mode: str,
) -> ParallelTrajectoryModel:
    return ParallelTrajectoryModel(
        horizon=horizon,
        history_mean=history_mean,
        history_scale=history_scale,
        control_mean=control_mean,
        control_scale=control_scale,
        edges=edges,
        gate_mode=gate_mode,
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
