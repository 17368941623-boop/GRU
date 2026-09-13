"""Raw54 history GRU plus the same untransformed future-control GRU used by 0908."""

from __future__ import annotations

import torch
from torch import nn


class Raw54GRU(nn.Module):
    def __init__(
        self,
        raw_count: int,
        control_count: int,
        horizon: int,
        history_hidden: int = 64,
        control_hidden: int = 32,
        fusion_hidden: int = 96,
        horizon_embedding: int = 8,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.history_gru = nn.GRU(raw_count, history_hidden, batch_first=True)
        self.control_gru = nn.GRU(control_count, control_hidden, batch_first=True)
        self.horizon_embedding = nn.Embedding(horizon, horizon_embedding)
        self.head = nn.Sequential(
            nn.Linear(history_hidden + control_hidden + horizon_embedding, fusion_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.SiLU(),
            nn.Linear(fusion_hidden // 2, 1),
        )

    def forward(self, history: torch.Tensor, future_controls: torch.Tensor) -> torch.Tensor:
        if future_controls.shape[1] != self.horizon:
            raise ValueError(f"Expected {self.horizon} future-control steps")
        _, history_state = self.history_gru(history)
        history_vector = history_state[-1]
        control_prefix, _ = self.control_gru(future_controls)
        horizon_ids = torch.arange(self.horizon, device=history.device)
        horizon_vector = self.horizon_embedding(horizon_ids)[None, :, :]
        horizon_vector = horizon_vector.expand(history.shape[0], -1, -1)
        history_vector = history_vector[:, None, :].expand(-1, self.horizon, -1)
        return self.head(
            torch.cat((history_vector, control_prefix, horizon_vector), dim=-1)
        ).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
