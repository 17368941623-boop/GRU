"""Compile the approved static topology and PCMCI results into model constraints.

The resulting JSON is a frozen training input.  Validation/test data are never
read here.  A second mode can rebuild the files from the saved source snapshot.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "2.0"
SAMPLE_PERIOD_SECONDS = 10
TARGET = "Thv"
H_MIX = "H_mix"
MAX_GENERIC_LAG = 30
MAX_PCMCI_LAG = 60
CONTROL_HISTORY_MAX_LAG = 59


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _number(value: str) -> int | float | str | bool:
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if numeric.is_integer() and "e" not in value.lower() and "." not in value:
        return int(numeric)
    return numeric


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{key: _number(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def load_static_relations(script_path: Path) -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("approved_static_graph", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import static graph source: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    relations = module.build_relations()
    if not isinstance(relations, list) or not relations:
        raise ValueError("Static graph source returned no relations")
    return relations


def infer_category(signal: str, primary_valves: set[str]) -> str:
    if signal in primary_valves or signal.startswith("CV") or signal in {"EC-V2", "COOLDOWN", "FC-V1"}:
        return "valve_or_control"
    if signal.startswith("PT"):
        return "pressure"
    if signal.startswith("FT"):
        return "flow"
    return "temperature"


def evidence_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "lag_steps": int(row["lag_steps"]),
        "lag_seconds": int(row["lag_seconds"]),
        "effect": float(row["effect"]),
        "q_value": float(row["q_value"]),
        "bootstrap_stability": float(row["bootstrap_stability"]),
        "selection_score": float(row["selection_score"]),
    }


def evidence_index(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    indexed: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        indexed[(str(row["source"]), str(row["destination"]))].append(evidence_record(row))
    for values in indexed.values():
        values.sort(key=lambda item: item["lag_steps"])
    return indexed


def lag_values(evidence: list[dict[str, Any]]) -> list[int]:
    return sorted({int(item["lag_steps"]) for item in evidence})


def compact_prior(lag_min: int, lag_max: int) -> list[int]:
    """Keep the full physical window; the model learns a sparse attention mask."""
    return list(range(int(lag_min), int(lag_max) + 1))


def make_snapshot(
    processed_data: Path,
    pcmci_root: Path,
    static_graph_script: Path,
) -> dict[str, Any]:
    catalog = read_json(processed_data / "feature_catalog.json")
    return {
        "schema_version": SCHEMA_VERSION,
        "source_policy": {
            "pcmci_input": "Original training parent runs only",
            "validation_or_test_used": False,
            "engineered_features_used_by_pcmci": False,
            "sample_period_seconds": SAMPLE_PERIOD_SECONDS,
        },
        "feature_catalog": catalog,
        "static_relations": load_static_relations(static_graph_script),
        "pcmci": {
            "difference_physical": read_csv(pcmci_root / "difference" / "stable_physical_lags.csv"),
            "level_physical": read_csv(pcmci_root / "level" / "stable_physical_lags.csv"),
            "difference_all": read_csv(pcmci_root / "difference" / "stable_pcmci_edges_all.csv"),
            "level_all": read_csv(pcmci_root / "level" / "stable_pcmci_edges_all.csv"),
            "hmix_difference": read_csv(
                pcmci_root / "hmix_targeted" / "difference" / "hmix_targeted_stable_lags.csv"
            ),
            "hmix_level": read_csv(
                pcmci_root / "hmix_targeted" / "level" / "hmix_targeted_stable_lags.csv"
            ),
            "difference_audit": read_json(pcmci_root / "difference" / "discovery_audit.json"),
            "level_audit": read_json(pcmci_root / "level" / "discovery_audit.json"),
            "transform_comparison": read_json(pcmci_root / "difference_vs_level_summary.json"),
        },
        "human_approval": {
            "approved_date": "2026-09-13",
            "status": "accepted_by_operator",
            "decisions": {
                "A管_to_Thv_effective_lag_steps": [13, 14, 15, 16, 17],
                "COOLDOWN_to_Thv_effective_lag_steps": [9, 10, 11, 12, 13],
                "EC-V2": "do not assert one hard physical lag; encode current level and learned past-action history",
                "FC-V1": "do not assert one hard physical lag; encode current level and learned past-action history",
                "H_mix": "logical virtual state; compile as one A管-to-Thv KAN conditioned edge",
            },
        },
    }


def select_optional_edges(
    rows: list[dict[str, Any]],
    existing_pairs: set[tuple[str, str]],
    active_controls: set[str],
    node_index: dict[str, int],
    limit: int = 32,
) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        source, destination = str(row["source"]), str(row["destination"])
        pair = (source, destination)
        if source == destination or pair in existing_pairs or destination in active_controls:
            continue
        if bool(row.get("physical_candidate", False)):
            continue
        if float(row["q_value"]) > 0.05 or float(row["bootstrap_stability"]) < 0.80:
            continue
        if float(row["abs_effect"]) < 0.03:
            continue
        if source not in node_index or destination not in node_index:
            continue
        if pair not in best or float(row["selection_score"]) > float(best[pair]["selection_score"]):
            best[pair] = row
    ranked = sorted(best.values(), key=lambda item: float(item["selection_score"]), reverse=True)[:limit]
    return [
        {
            "edge_id": f"optional_pcmci_{rank + 1:02d}",
            "source": str(row["source"]),
            "destination": str(row["destination"]),
            "source_index": node_index[str(row["source"])],
            "destination_index": node_index[str(row["destination"])],
            "relation_type": "optional_data_driven",
            "enabled_by_default": False,
            "representation": "difference",
            "lag_steps": [int(row["lag_steps"])],
            "lag_seconds": [int(row["lag_seconds"])],
            "effect": float(row["effect"]),
            "q_value": float(row["q_value"]),
            "bootstrap_stability": float(row["bootstrap_stability"]),
            "selection_score": float(row["selection_score"]),
            "direct_to_target": str(row["destination"]) == TARGET,
            "use_rule": "only in the optional-PCMCI ablation and with a learnable gate",
        }
        for rank, row in enumerate(ranked)
    ]


def build_outputs(snapshot: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    feature_catalog = snapshot["feature_catalog"]
    raw_columns = list(feature_catalog["raw_signal_columns"])
    if len(raw_columns) != 54 or len(set(raw_columns)) != 54:
        raise ValueError("Expected 54 unique Raw54 signals")
    node_index = {name: index for index, name in enumerate(raw_columns)}
    active_controls = set(feature_catalog["primary_valve_columns"])
    nodes = [
        {
            "node_index": index,
            "signal": signal,
            "category": infer_category(signal, active_controls),
            "is_target": signal == TARGET,
            "is_control": signal in active_controls,
            "is_observed": True,
        }
        for index, signal in enumerate(raw_columns)
    ]
    logical_nodes = nodes + [
        {
            "node_index": 54,
            "signal": H_MIX,
            "category": "latent_thermal_hydraulic_state",
            "is_target": False,
            "is_control": False,
            "is_observed": False,
        }
    ]

    diff_idx = evidence_index(snapshot["pcmci"]["difference_physical"])
    level_idx = evidence_index(snapshot["pcmci"]["level_physical"])
    generic_edges: list[dict[str, Any]] = []
    collapsed_relations: list[dict[str, Any]] = []

    hmix_relation_pairs = {
        ("A管", H_MIX),
        ("EC-V2", H_MIX),
        ("COOLDOWN", H_MIX),
        ("FC-V1", H_MIX),
        (H_MIX, TARGET),
    }
    for relation in snapshot["static_relations"]:
        source, destination = str(relation["source"]), str(relation["destination"])
        pair = (source, destination)
        if pair in hmix_relation_pairs:
            collapsed_relations.append(relation)
            continue
        if source not in node_index or destination not in node_index:
            raise KeyError(f"Static relation uses an unknown observed node: {pair}")
        diff_evidence = diff_idx.get(pair, [])
        level_evidence = level_idx.get(pair, [])
        source_category = nodes[node_index[source]]["category"]
        source_is_valve = source_category == "valve_or_control"
        fallback = not diff_evidence and not level_evidence

        if diff_evidence:
            difference_pool = lag_values(diff_evidence)
        elif level_evidence:
            difference_pool = []
        else:
            difference_pool = compact_prior(int(relation["lag_min"]), int(relation["lag_max"]))

        level_pool = lag_values(level_evidence)
        if source_is_valve:
            level_pool = sorted(set([0, *level_pool]))

        if diff_evidence:
            active_representations = ["difference"]
            evidence_class = "difference_primary"
        elif level_evidence:
            active_representations = ["level"]
            evidence_class = "level_only_sensitivity_support"
        else:
            active_representations = ["difference"]
            evidence_class = "physical_prior_only"
        if source_is_valve:
            active_representations = ["level", "difference"]
            if not difference_pool:
                difference_pool = compact_prior(int(relation["lag_min"]), int(relation["lag_max"]))

        role = "self_memory" if source == destination else "static_physical"
        edge_function_id = relation.get("edge_function_id", "shared_lag_message")
        if relation.get("edge_function_mode") == "KAN_dynamic_branch_gate":
            role = "three_branch_kan_governed"

        generic_edges.append(
            {
                "edge_id": f"physical_{len(generic_edges):03d}",
                "source": source,
                "destination": destination,
                "source_index": node_index[source],
                "destination_index": node_index[destination],
                "relation_type": role,
                "relation_group": relation["relation_group"],
                "mechanism": relation["mechanism"],
                "physical_confidence": relation["confidence"],
                "edge_function_id": edge_function_id,
                "active_representations": active_representations,
                "lag_pools": {
                    "difference": difference_pool,
                    "level": level_pool,
                },
                "lag_selection": {
                    "strategy": "attention_over_allowed_pool",
                    "evidence_class": evidence_class,
                    "fallback_to_physical_window": fallback,
                    "hard_single_lag": False,
                },
                "statistical_evidence": {
                    "difference": diff_evidence,
                    "level": level_evidence,
                },
            }
        )

    hmix_diff_idx = evidence_index(snapshot["pcmci"]["hmix_difference"])
    hmix_level_idx = evidence_index(snapshot["pcmci"]["hmix_level"])
    hmix_edge = {
        "edge_id": f"physical_{len(generic_edges):03d}",
        "source": "A管",
        "destination": TARGET,
        "source_index": node_index["A管"],
        "destination_index": node_index[TARGET],
        "relation_type": "hmix_conditioned_edge",
        "edge_function_id": "downstream_hmix_kan",
        "enabled_by_default": True,
        "compiled_from_logical_path": [
            "A管/EC-V2/COOLDOWN/FC-V1 -> H_mix",
            "H_mix -> Thv",
        ],
        "active_representations": ["level", "difference"],
        "lag_pools": {"level": [0], "difference": [13, 14, 15, 16, 17]},
        "lag_selection": {
            "strategy": "attention_over_allowed_pool",
            "evidence_class": "human_approved_targeted_pcmci_consensus",
            "hard_single_lag": False,
        },
        "condition_inputs": {
            "A管": {
                "level_lags": [0],
                "difference_lags": [13, 14, 15, 16, 17],
                "encoder": "lag_attention",
                "physical_lag_claim": "130-170 s effective response window",
            },
            "COOLDOWN": {
                "level_lags": [0],
                "difference_lags": [9, 10, 11, 12, 13],
                "encoder": "lag_attention",
                "physical_lag_claim": "90-130 s effective response window",
            },
            "EC-V2": {
                "level_lags": [0],
                "difference_lags": list(range(0, CONTROL_HISTORY_MAX_LAG + 1)),
                "encoder": "control_history_gru",
                "physical_lag_claim": None,
                "reason": "targeted PCMCI lag was boundary-sensitive and confounded with COOLDOWN",
            },
            "FC-V1": {
                "level_lags": [0],
                "difference_lags": list(range(0, CONTROL_HISTORY_MAX_LAG + 1)),
                "encoder": "control_history_gru",
                "physical_lag_claim": None,
                "reason": "no stable independent targeted PCMCI lag",
            },
        },
        "statistical_evidence": {
            "A管_difference": hmix_diff_idx.get(("A管", TARGET), []),
            "A管_level": hmix_level_idx.get(("A管", TARGET), []),
            "COOLDOWN_difference": hmix_diff_idx.get(("COOLDOWN", TARGET), []),
            "EC-V2_difference": hmix_diff_idx.get(("EC-V2", TARGET), []),
            "FC-V1_difference": hmix_diff_idx.get(("FC-V1", TARGET), []),
        },
    }
    generic_edges.append(hmix_edge)

    existing_pairs = {(edge["source"], edge["destination"]) for edge in generic_edges}
    optional_edges = select_optional_edges(
        snapshot["pcmci"]["difference_all"], existing_pairs, active_controls, node_index
    )
    edge_index = [
        [edge["source_index"] for edge in generic_edges],
        [edge["destination_index"] for edge in generic_edges],
    ]

    branch_kan = {
        "function_id": "three_branch_kan_gate",
        "model": "KAN",
        "branch_history_steps": 60,
        "branch_inputs": {
            "branch_1": ["CV8312", "CV8330", "TE8330", "PT8330"],
            "branch_2": ["CV8311", "CV8350", "TE8350", "PT8350"],
            "branch_3": ["CV8310", "CV8351", "TE8310", "PT8310"],
        },
        "input_policy": {
            "valves": ["level", "difference"],
            "temperature_pressure_flow": ["difference"],
        },
        "outputs": [
            {"name": "alpha_T", "governs": "TE8310 -> TE8351"},
            {"name": "alpha_P", "governs": "PT8310 -> PT8351"},
        ],
        "interpretation": "branch encodings condition two physical messages; they are not extra material-flow edges",
    }

    graph = {
        "schema_version": SCHEMA_VERSION,
        "name": "HTC8300 approved physical and PCMCI lag constraints",
        "status": "frozen_for_model_ablation_v2",
        "sample_period_seconds": SAMPLE_PERIOD_SECONDS,
        "history_steps": 60,
        "raw_rows_required_for_history": 61,
        "forecast_steps": 30,
        "forecast_seconds": 300,
        "target": TARGET,
        "input_semantics": {
            "global_gru": "all Raw54 levels for t-59..t (60 values)",
            "difference_history": "60 differences for t-59..t, computed causally from raw rows t-60..t",
            "process_gnn": "temperature/pressure/flow differences by default; level only where explicitly supported",
            "valve_gnn": "current/past opening level plus opening difference",
            "target_decoder": "predict 30 future Thv increments and add Thv(t) cumulatively",
        },
        "logical_graph": {
            "nodes": logical_nodes,
            "contains_virtual_H_mix": True,
            "virtual_topology": "A管/EC-V2/COOLDOWN/FC-V1 -> H_mix -> Thv",
        },
        "compiled_graph": {
            "nodes": nodes,
            "edge_index": edge_index,
            "edges": generic_edges,
            "H_mix_compilation": "collapsed conditioned edge A管 -> Thv; the three valves are function inputs",
            "self_memory_edges_enabled_by_default": True,
        },
        "special_functions": {
            "three_branch_kan": branch_kan,
            "downstream_hmix_kan_edge_id": hmix_edge["edge_id"],
        },
        "optional_pcmci_edges": optional_edges,
        "optional_pcmci_edges_enabled_by_default": False,
        "training_guards": {
            "lag_discovery_split": "Original training parent runs only",
            "reuse_same_graph_for_all_seeds": True,
            "validation_may_select_model_not_graph": True,
            "frozen_test_access_before_final_selection": False,
            "future_process_measurements_allowed": False,
            "future_controls_allowed_only_if_available_at_prediction_time": True,
            "primary_control_mode": "history_only",
            "secondary_control_mode": "known_plan_only",
            "recorded_future_valves_must_not_be_used_as_if_known": True,
        },
        "human_approval": snapshot["human_approval"],
    }

    ablations = {
        "schema_version": SCHEMA_VERSION,
        "fair_comparison_contract": {
            "forecast_steps": 30,
            "lookback_steps": 60,
            "same_parent_split": True,
            "same_scalers_and_loss": True,
            "same_seed_list": [42, 52, 62, 72, 82, 92, 102, 112, 122, 132],
            "note": "The 0912 Raw54-GRU result used 15 steps and must be rerun at 30 steps for the final five-minute table.",
        },
        "control_modes": {
            "primary": {
                "name": "history_only",
                "future_control_sequence": False,
                "reason": "matches deployment when future valve openings are unknown",
            },
            "secondary": {
                "name": "known_plan_only",
                "future_control_sequence": True,
                "reason": "for MPC candidate trajectories supplied before prediction; never use recorded future values as a proxy for a plan",
            },
            "comparison_rule": "all models in one table must use the same control mode",
        },
        "models": [
            {"id": "A0", "name": "Raw54-GRU-5min", "global_gru": True, "mixed_rep": False, "static_gnn": False, "pcmci_lags": False, "branch_kan": False, "hmix_kan": False, "optional_pcmci": False},
            {"id": "A1", "name": "MixedRep-GRU", "global_gru": True, "mixed_rep": True, "static_gnn": False, "pcmci_lags": False, "branch_kan": False, "hmix_kan": False, "optional_pcmci": False},
            {"id": "A2", "name": "Static-GNN+GRU", "global_gru": True, "mixed_rep": True, "static_gnn": True, "pcmci_lags": False, "branch_kan": False, "hmix_kan": False, "optional_pcmci": False},
            {"id": "A3", "name": "PCMCI-Lag-GNN+GRU", "global_gru": True, "mixed_rep": True, "static_gnn": True, "pcmci_lags": True, "branch_kan": False, "hmix_kan": False, "optional_pcmci": False},
            {"id": "A4", "name": "PCMCI-Lag-GNN+BranchKAN+GRU", "global_gru": True, "mixed_rep": True, "static_gnn": True, "pcmci_lags": True, "branch_kan": True, "hmix_kan": False, "optional_pcmci": False},
            {"id": "A5", "name": "PCMCI-Lag-GNN+HmixKAN+GRU", "global_gru": True, "mixed_rep": True, "static_gnn": True, "pcmci_lags": True, "branch_kan": False, "hmix_kan": True, "optional_pcmci": False},
            {"id": "A6", "name": "Parallel-GNN+KAN+GRU", "global_gru": True, "mixed_rep": True, "static_gnn": True, "pcmci_lags": True, "branch_kan": True, "hmix_kan": True, "optional_pcmci": False},
            {"id": "A7", "name": "Parallel-GNN+KAN+GRU+OptionalPCMCI", "global_gru": True, "mixed_rep": True, "static_gnn": True, "pcmci_lags": True, "branch_kan": True, "hmix_kan": True, "optional_pcmci": True},
        ],
        "execution_stages": [
            {"stage": "smoke", "models": ["A0", "A2", "A6"], "seeds": [42], "purpose": "shape, leakage and convergence checks"},
            {"stage": "screen", "models": ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"], "seeds": [42, 72, 102], "purpose": "screen modules before expensive repeats"},
            {"stage": "final", "models": ["A0", "A3", "A4", "A5", "A6"], "seeds": [42, 52, 62, 72, 82, 92, 102, 112, 122, 132], "purpose": "paper table and seed-distribution inference"},
        ],
        "required_outputs": [
            "overall RMSE, MAE and R2 per seed",
            "horizon RMSE/MAE at 50, 100, 150, 200 and 300 s",
            "rapid-cooling subset and valve-event subset metrics",
            "seed mean, standard deviation and 95% confidence interval",
            "paired per-window errors against A0 on the same validation samples",
            "parameter count, training time and peak GPU memory",
            "learned lag-attention weights and KAN response curves",
            "full validation predictions and prediction indices",
        ],
    }

    evidence_classes = defaultdict(int)
    relation_types = defaultdict(int)
    for edge in generic_edges:
        evidence_classes[edge["lag_selection"]["evidence_class"]] += 1
        relation_types[edge["relation_type"]] += 1
    audit = {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "observed_nodes": len(nodes),
        "logical_nodes_including_H_mix": len(logical_nodes),
        "compiled_edges": len(generic_edges),
        "compiled_cross_edges": sum(edge["source"] != edge["destination"] for edge in generic_edges),
        "compiled_self_edges": sum(edge["source"] == edge["destination"] for edge in generic_edges),
        "relation_type_counts": dict(sorted(relation_types.items())),
        "lag_evidence_class_counts": dict(sorted(evidence_classes.items())),
        "optional_pcmci_edges": len(optional_edges),
        "static_H_mix_relations_collapsed": len(collapsed_relations),
        "manual_lag_decisions": snapshot["human_approval"]["decisions"],
        "maximum_generic_lag_steps": max(
            [0]
            + [lag for edge in generic_edges for values in edge["lag_pools"].values() for lag in values]
        ),
        "maximum_condition_history_lag_steps": CONTROL_HISTORY_MAX_LAG,
        "targeted_pcmci_search_max_lag_steps": MAX_PCMCI_LAG,
        "notes": [
            "PCMCI values are allowed lag pools, not hard-coded single causal delays.",
            "Physical-prior-only edges remain in the static graph and learn attention inside their approved search window.",
            "Optional nonphysical PCMCI edges are frozen off unless the A7 ablation is selected.",
        ],
    }
    return graph, ablations, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--processed-data", type=Path)
    parser.add_argument("--pcmci-root", type=Path)
    parser.add_argument("--static-graph-script", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.snapshot:
        snapshot = read_json(args.snapshot)
    else:
        missing = [
            name
            for name, value in (
                ("--processed-data", args.processed_data),
                ("--pcmci-root", args.pcmci_root),
                ("--static-graph-script", args.static_graph_script),
            )
            if value is None
        ]
        if missing:
            raise SystemExit("Missing inputs: " + ", ".join(missing))
        snapshot = make_snapshot(args.processed_data, args.pcmci_root, args.static_graph_script)

    graph, ablations, audit = build_outputs(snapshot)
    output_dir = args.output_dir.resolve()
    write_json(output_dir / "source_snapshot_v2.json", snapshot)
    write_json(output_dir / "graph_constraints_v2.json", graph)
    write_json(output_dir / "ablation_matrix_v2.json", ablations)
    write_json(output_dir / "graph_constraint_audit_v2.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
