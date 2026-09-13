"""Fail-fast validation for the frozen graph package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from graph_constraints import build_graph_tensors, load_config


def validate(config: dict) -> dict:
    nodes = config["compiled_graph"]["nodes"]
    edges = config["compiled_graph"]["edges"]
    names = [node["signal"] for node in nodes]
    assert len(names) == 54 and len(set(names)) == 54
    assert names.index("Thv") == 48
    assert config["forecast_steps"] == 30
    assert config["forecast_seconds"] == 300
    assert config["history_steps"] == 60
    assert config["raw_rows_required_for_history"] == 61
    assert config["logical_graph"]["contains_virtual_H_mix"] is True
    assert "H_mix" not in names

    edge_ids = [edge["edge_id"] for edge in edges]
    assert len(edge_ids) == len(set(edge_ids))
    for edge in edges:
        assert names[edge["source_index"]] == edge["source"]
        assert names[edge["destination_index"]] == edge["destination"]
        for pool in edge["lag_pools"].values():
            assert all(0 <= int(lag) <= 60 for lag in pool)

    pairs = {(edge["source"], edge["destination"]): edge for edge in edges}
    assert ("TE8310", "TE8351") in pairs
    assert ("PT8310", "PT8351") in pairs
    assert pairs[("TE8310", "TE8351")]["relation_type"] == "three_branch_kan_governed"
    assert pairs[("PT8310", "PT8351")]["relation_type"] == "three_branch_kan_governed"

    hmix = pairs[("A管", "Thv")]
    assert hmix["relation_type"] == "hmix_conditioned_edge"
    assert hmix["condition_inputs"]["A管"]["difference_lags"] == [13, 14, 15, 16, 17]
    assert hmix["condition_inputs"]["COOLDOWN"]["difference_lags"] == [9, 10, 11, 12, 13]
    assert hmix["condition_inputs"]["EC-V2"]["physical_lag_claim"] is None
    assert hmix["condition_inputs"]["FC-V1"]["physical_lag_claim"] is None
    assert not any(pair in pairs for pair in (("EC-V2", "Thv"), ("COOLDOWN", "Thv"), ("FC-V1", "Thv")))

    controls = {node["signal"] for node in nodes if node["is_control"]}
    assert not any(edge["destination"] in controls and edge["source"] not in controls for edge in edges)
    assert config["optional_pcmci_edges_enabled_by_default"] is False

    tensors = build_graph_tensors(config)
    assert tuple(tensors["edge_index"].shape) == (2, len(edges))
    assert tuple(tensors["level_lag_mask"].shape) == (len(edges), 61)
    assert tuple(tensors["difference_lag_mask"].shape) == (len(edges), 61)
    return {
        "status": "PASS",
        "nodes": len(nodes),
        "edges": len(edges),
        "self_edges": sum(edge["source"] == edge["destination"] for edge in edges),
        "cross_edges": sum(edge["source"] != edge["destination"] for edge in edges),
        "optional_pcmci_edges": len(config["optional_pcmci_edges"]),
        "hmix_compilation": config["compiled_graph"]["H_mix_compilation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "graph_constraints_v2.json",
    )
    args = parser.parse_args()
    print(json.dumps(validate(load_config(args.config)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
