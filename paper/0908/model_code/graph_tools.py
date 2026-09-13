#!/usr/bin/env python3
"""Graph schema, physical-prior expansion and causal/domain graph variants."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from protocol import MAX_CAUSAL_LAG, RAW_COLUMNS


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_graph(path: Path, name: str, edges: list[dict[str, Any]], **metadata: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_edges(edges)
    node_index = {name: index for index, name in enumerate(RAW_COLUMNS)}
    payload = {
        "schema_version": 1,
        "name": name,
        "nodes": list(RAW_COLUMNS),
        "node_index": node_index,
        "edge_semantics": "source(t-lag_steps) -> destination(t)",
        "edge_count": len(edges),
        "edge_index": [
            [node_index[str(edge["source"])] for edge in edges],
            [node_index[str(edge["destination"])] for edge in edges],
        ],
        "edge_lag_steps": [int(edge["lag_steps"]) for edge in edges],
        "edges": edges,
        **metadata,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_edges(edges: list[dict[str, Any]], max_lag: int = MAX_CAUSAL_LAG) -> None:
    allowed = set(RAW_COLUMNS)
    seen: set[tuple[str, str, int]] = set()
    for edge in edges:
        source = str(edge["source"])
        destination = str(edge["destination"])
        lag = int(edge["lag_steps"])
        key = (source, destination, lag)
        if source not in allowed or destination not in allowed:
            raise ValueError(f"Unknown node in edge {key}")
        if not 1 <= lag <= max_lag:
            raise ValueError(f"Lag outside [1, {max_lag}] in edge {key}")
        if key in seen:
            raise ValueError(f"Duplicate edge {key}")
        seen.add(key)


def load_edges(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    edges = list(payload.get("edges", []))
    validate_edges(edges)
    return edges


def physical_edges(static_path: Path, all_lags: bool) -> list[dict[str, Any]]:
    relations = read_json(static_path)["relations"]
    edges: list[dict[str, Any]] = []
    for relation in relations:
        lags = (
            range(int(relation["lag_min"]), int(relation["lag_max"]) + 1)
            if all_lags else (int(relation["preferred_lag"]),)
        )
        for lag in lags:
            edges.append(
                {
                    "source": relation["source"],
                    "destination": relation["destination"],
                    "lag_steps": lag,
                    "lag_seconds": lag * 10,
                    "edge_type": "physical",
                    "confidence": relation["confidence"],
                    "mechanism": relation["mechanism"],
                    "expected_sign": relation["expected_sign"],
                    "forced_by_domain": True,
                }
            )
    validate_edges(edges)
    return edges


def _prior_lookup(static_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(item["source"]), str(item["destination"])): item
        for item in read_json(static_path)["relations"]
    }


def create_graph_variants(
    selected_edges: list[dict[str, Any]],
    static_path: Path,
    output_dir: Path,
    random_seed: int = 20260908,
) -> dict[str, int]:
    """Create the fixed comparison graphs after causal discovery is complete."""
    validate_edges(selected_edges)
    output_dir.mkdir(parents=True, exist_ok=True)
    prior = _prior_lookup(static_path)

    # The data-only control must not inherit candidates that entered the PCMCI
    # search solely because of the physical prior. Such edges are still valid
    # for DK&CDV/DK&CDL after they pass the same MCI and stability thresholds.
    cd = [
        dict(edge, edge_type="data", forced_by_domain=False)
        for edge in selected_edges
        if bool(edge.get("data_screened", True))
    ]
    if not cd:
        raise ValueError(
            "PCMCI selection produced no edges. Inspect pcmci_all_lag_tests.csv and relax "
            "min-effect/min-stability only from training evidence."
        )
    write_graph(output_dir / "cd_select_lag.json", "PCMCI stable selected-lag graph", cd)

    primary: list[dict[str, Any]] = []
    discovered_pairs: set[tuple[str, str]] = set()
    for original in selected_edges:
        edge = dict(original, edge_type="data", forced_by_domain=False)
        pair = (str(edge["source"]), str(edge["destination"]))
        discovered_pairs.add(pair)
        if pair in prior:
            relation = prior[pair]
            edge = dict(
                edge,
                edge_type="both",
                physical_confidence=relation["confidence"],
                mechanism=relation["mechanism"],
                expected_sign=relation["expected_sign"],
            )
        primary.append(edge)
    for pair, relation in prior.items():
        if pair in discovered_pairs:
            continue
        lag = int(relation["preferred_lag"])
        primary.append(
            {
                "source": pair[0], "destination": pair[1], "lag_steps": lag,
                "lag_seconds": lag * 10, "edge_type": "physical",
                "confidence": relation["confidence"], "confidence_score": 0.40,
                "mechanism": relation["mechanism"],
                "expected_sign": relation["expected_sign"],
                "forced_by_domain": True, "effect": None, "p_value": None,
                "q_value": None, "bootstrap_stability": None,
            }
        )
    validate_edges(primary)
    write_graph(
        output_dir / "dkcdv_select_lag.json",
        "Domain-validated PCMCI graph (primary)", primary,
        note="Union of stable PCMCI edges and one preferred-lag edge for each unsupported documented pair.",
    )

    limited = []
    for original in selected_edges:
        edge = dict(original, edge_type="data", forced_by_domain=False)
        pair = (str(edge["source"]), str(edge["destination"]))
        if pair in prior:
            relation = prior[pair]
            limited.append(
                dict(
                    edge, edge_type="both", physical_confidence=relation["confidence"],
                    mechanism=relation["mechanism"], expected_sign=relation["expected_sign"],
                )
            )
    validate_edges(limited)
    if not limited:
        raise ValueError(
            "No stable PCMCI edge overlaps the physical prior; the DK&CDL comparison would be empty."
        )
    write_graph(
        output_dir / "dkcdl_select_lag.json",
        "Domain-limited PCMCI graph", limited,
        note="Intersection: stable PCMCI edges whose variable pair occurs in the physical prior.",
    )

    full_physical = physical_edges(static_path, all_lags=True)
    write_graph(
        output_dir / "physical_all_lags.json", "Physical prior expanded across lag bands", full_physical
    )

    rng = random.Random(random_seed)
    used_by_pair: dict[tuple[str, str], set[int]] = {}
    shuffled = []
    for edge in primary:
        pair = (str(edge["source"]), str(edge["destination"]))
        available = [lag for lag in range(1, MAX_CAUSAL_LAG + 1) if lag not in used_by_pair.setdefault(pair, set())]
        original_lag = int(edge["lag_steps"])
        alternatives = [lag for lag in available if lag != original_lag]
        lag = rng.choice(alternatives or available)
        used_by_pair[pair].add(lag)
        shuffled.append(
            dict(edge, lag_steps=lag, lag_seconds=lag * 10, edge_type=edge.get("edge_type", "both"))
        )
    validate_edges(shuffled)
    write_graph(
        output_dir / "dkcdv_shuffled_lag.json", "Primary variable pairs with permuted lags", shuffled,
        random_seed=random_seed,
    )

    candidates = [
        (source, destination, lag)
        for source in RAW_COLUMNS for destination in RAW_COLUMNS
        for lag in range(1, MAX_CAUSAL_LAG + 1)
    ]
    rng.shuffle(candidates)
    random_edges = [
        {
            "source": source, "destination": destination, "lag_steps": lag,
            "lag_seconds": lag * 10, "edge_type": "random", "confidence": "low",
            "confidence_score": 0.25, "forced_by_domain": False,
        }
        for source, destination, lag in candidates[: len(primary)]
    ]
    validate_edges(random_edges)
    write_graph(
        output_dir / "random_same_size.json", "Random same-size negative-control graph",
        random_edges, random_seed=random_seed, matched_edge_count=len(primary),
    )
    return {
        "cd_select_lag": len(cd),
        "dkcdv_select_lag": len(primary),
        "dkcdl_select_lag": len(limited),
        "physical_all_lags": len(full_physical),
        "dkcdv_shuffled_lag": len(shuffled),
        "random_same_size": len(random_edges),
    }


def bootstrap_physical_graphs(static_path: Path, output_dir: Path) -> None:
    """Create only the static graph so preflight can run before PCMCI."""
    edges = physical_edges(static_path, all_lags=True)
    write_graph(output_dir / "physical_all_lags.json", "Physical prior expanded across lag bands", edges)
