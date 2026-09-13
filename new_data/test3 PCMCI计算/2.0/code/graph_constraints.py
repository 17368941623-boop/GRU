"""Load graph_constraints_v2.json and build tensors for a lag-aware GNN."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_graph_tensors(
    config: dict[str, Any],
    *,
    include_self_memory: bool = True,
    include_optional_pcmci: bool = False,
    max_lag_steps: int = 60,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Return unique node-pair edges and representation-specific lag masks.

    A mask value at column k means read source(t-k).  For differences, column
    zero means source(t)-source(t-1), which is known at the prediction origin.
    """
    edges = list(config["compiled_graph"]["edges"])
    if not include_self_memory:
        edges = [edge for edge in edges if edge["source"] != edge["destination"]]
    if include_optional_pcmci:
        for edge in config["optional_pcmci_edges"]:
            edges.append(
                {
                    **edge,
                    "active_representations": [edge["representation"]],
                    "lag_pools": {edge["representation"]: edge["lag_steps"]},
                    "edge_function_id": "gated_optional_pcmci_message",
                }
            )

    edge_index = torch.tensor(
        [[edge["source_index"] for edge in edges], [edge["destination_index"] for edge in edges]],
        dtype=torch.long,
        device=device,
    )
    level_mask = torch.zeros((len(edges), max_lag_steps + 1), dtype=torch.bool, device=device)
    difference_mask = torch.zeros_like(level_mask)
    for edge_number, edge in enumerate(edges):
        for representation, mask in (("level", level_mask), ("difference", difference_mask)):
            for lag in edge.get("lag_pools", {}).get(representation, []):
                lag = int(lag)
                if lag < 0 or lag > max_lag_steps:
                    raise ValueError(f"{edge['edge_id']} has lag {lag} outside 0..{max_lag_steps}")
                mask[edge_number, lag] = True

    relation_names = sorted({edge["relation_type"] for edge in edges})
    relation_lookup = {name: index for index, name in enumerate(relation_names)}
    relation_id = torch.tensor(
        [relation_lookup[edge["relation_type"]] for edge in edges],
        dtype=torch.long,
        device=device,
    )
    return {
        "node_names": [node["signal"] for node in config["compiled_graph"]["nodes"]],
        "edges": edges,
        "edge_index": edge_index,
        "level_lag_mask": level_mask,
        "difference_lag_mask": difference_mask,
        "relation_id": relation_id,
        "relation_lookup": relation_lookup,
        "edge_function_ids": [edge["edge_function_id"] for edge in edges],
        "hmix_edge_id": config["special_functions"]["downstream_hmix_kan_edge_id"],
    }


def gather_edge_lag_values(
    history: torch.Tensor,
    edge_index: torch.Tensor,
    lag_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather every edge source at t, t-1, ..., t-max_lag.

    Parameters
    ----------
    history:
        Shape [batch, time, node, feature], ending at prediction origin t.
    edge_index:
        Shape [2, edge].  Only the source row is used for gathering.
    lag_mask:
        Shape [edge, max_lag+1].  Column k means source(t-k).

    Returns
    -------
    values, mask:
        Values have shape [batch, edge, max_lag+1, feature].  The returned
        mask is broadcastable to the same shape and should be applied before
        lag attention.  No index after t is constructed.
    """
    if history.ndim != 4:
        raise ValueError("history must have shape [batch, time, node, feature]")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edge]")
    if lag_mask.ndim != 2 or lag_mask.shape[0] != edge_index.shape[1]:
        raise ValueError("lag_mask must have shape [edge, max_lag+1]")
    active = torch.nonzero(lag_mask, as_tuple=False)
    if active.numel() == 0:
        raise ValueError("lag_mask contains no allowed lags")
    max_lag = int(active[:, 1].max().item())
    if max_lag >= history.shape[1]:
        raise ValueError("history is shorter than max_lag+1")

    batch, _, _, features = history.shape
    edge_count = edge_index.shape[1]
    time_index = torch.arange(
        history.shape[1] - 1,
        history.shape[1] - 2 - max_lag,
        -1,
        device=history.device,
    )
    lagged = history.index_select(1, time_index)
    source_index = edge_index[0].to(history.device)
    gather_index = source_index.view(1, 1, edge_count, 1).expand(
        batch, max_lag + 1, edge_count, features
    )
    values = torch.gather(lagged, dim=2, index=gather_index).permute(0, 2, 1, 3)
    effective_mask = lag_mask[:, : max_lag + 1]
    return values, effective_mask.to(history.device).view(1, edge_count, max_lag + 1, 1)
