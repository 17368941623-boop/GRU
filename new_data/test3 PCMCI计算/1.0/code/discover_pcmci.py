#!/usr/bin/env python3
"""Raw54 lag discovery: independent screening + restricted PCMCI + bootstrap."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_io import (
    bootstrap_parent_runs,
    load_original_training_records,
    split_parent_runs,
    transform_records,
)
from protocol import (
    ACTIVE_CONTROLS,
    DEFAULT_MAX_LAG,
    SAMPLE_PERIOD_SECONDS,
    expand_static_candidates,
    load_raw_columns,
)


Triple = tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    code_dir = Path(__file__).resolve().parent
    experiment_dir = code_dir.parent
    data_dir = experiment_dir.parent / "processed_data"
    parser = argparse.ArgumentParser(
        description="Leakage-safe Raw54 PCMCI lag discovery for the HTC8300 process."
    )
    parser.add_argument("--data-dir", type=Path, default=data_dir)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=experiment_dir / "output" / "difference")
    parser.add_argument("--transform", choices=("difference", "level"), default="difference")
    parser.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG)
    parser.add_argument("--screen-per-destination", type=int, default=30)
    parser.add_argument("--screen-min-correlation", type=float, default=0.02)
    parser.add_argument("--pc-alpha", type=float, default=0.05)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap-candidate-q", type=float, default=0.10)
    parser.add_argument("--bootstrap", type=int, default=50)
    parser.add_argument("--min-stability", type=float, default=0.60)
    parser.add_argument("--min-effect", type=float, default=0.03)
    parser.add_argument("--top-lags-per-pair", type=int, default=3)
    parser.add_argument(
        "--max-bootstrap-candidates",
        type=int,
        default=0,
        help="0 means bootstrap every q/effect-qualified lag after the per-pair limit.",
    )
    parser.add_argument(
        "--max-export-edges",
        type=int,
        default=200,
        help="Compact graph export limit; does not limit statistical bootstrap.",
    )
    parser.add_argument(
        "--max-total-edges",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--max-conds-dim", type=int, default=10)
    parser.add_argument("--max-samples-per-segment", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--skip-stationarity", action="store_true")
    parser.add_argument("--verbosity", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.max_lag <= DEFAULT_MAX_LAG:
        raise ValueError(f"max-lag must be in [1, {DEFAULT_MAX_LAG}]")
    if args.screen_per_destination < 1 or args.max_conds_dim < 1:
        raise ValueError("screening and conditioning limits must be positive")
    if not 1 <= args.top_lags_per_pair <= args.max_lag:
        raise ValueError("top-lags-per-pair must be between 1 and max-lag")
    if args.max_total_edges is not None:
        warnings.warn(
            "--max-total-edges is deprecated; applying it to both bootstrap and export limits",
            FutureWarning,
        )
        args.max_bootstrap_candidates = args.max_total_edges
        args.max_export_edges = args.max_total_edges
    if args.max_bootstrap_candidates < 0 or args.max_export_edges < 1 or args.bootstrap < 0:
        raise ValueError(
            "max-bootstrap-candidates must be non-negative, max-export-edges positive, "
            "and bootstrap non-negative"
        )
    for name in ("pc_alpha", "fdr_alpha", "bootstrap_candidate_q", "min_stability"):
        value = float(getattr(args, name))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    values = np.where(np.isfinite(values), values, 1.0)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0.0, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def lagged_correlation_screen(
    segments: dict[int, np.ndarray], raw_columns: tuple[str, ...], max_lag: int
) -> pd.DataFrame:
    """Compute every source x destination x lag marginal Pearson correlation."""
    variable_count = len(raw_columns)
    source_indices = np.repeat(np.arange(variable_count), variable_count)
    destination_indices = np.tile(np.arange(variable_count), variable_count)
    blocks: list[pd.DataFrame] = []
    for lag in range(1, max_lag + 1):
        valid_segments = [values for values in segments.values() if len(values) > lag]
        source_values = np.concatenate([values[:-lag] for values in valid_segments], axis=0)
        destination_values = np.concatenate([values[lag:] for values in valid_segments], axis=0)
        source_centered = source_values - source_values.mean(axis=0)
        destination_centered = destination_values - destination_values.mean(axis=0)
        numerator = source_centered.T @ destination_centered
        denominator = np.sqrt(
            np.sum(source_centered ** 2, axis=0)[:, None]
            * np.sum(destination_centered ** 2, axis=0)[None, :]
        )
        correlation = np.divide(
            numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12
        )
        flat = correlation.reshape(-1)
        blocks.append(
            pd.DataFrame(
                {
                    "source": [raw_columns[index] for index in source_indices],
                    "destination": [raw_columns[index] for index in destination_indices],
                    "source_index": source_indices,
                    "destination_index": destination_indices,
                    "lag_steps": lag,
                    "lag_seconds": lag * SAMPLE_PERIOD_SECONDS,
                    "screen_correlation": flat,
                    "abs_screen_correlation": np.abs(flat),
                    "screen_sample_pairs": len(source_values),
                }
            )
        )
    result = pd.concat(blocks, ignore_index=True)
    expected = variable_count * variable_count * max_lag
    if len(result) != expected:
        raise AssertionError(f"Expected {expected} lagged correlations, got {len(result)}")
    return result


def select_candidates(
    screen: pd.DataFrame,
    raw_columns: tuple[str, ...],
    physical: set[Triple],
    physical_metadata: dict[Triple, dict[str, Any]],
    per_destination: int,
    min_correlation: float,
) -> tuple[set[Triple], set[Triple], pd.DataFrame]:
    # Controls are interventions/exogenous inputs: do not learn process -> control edges.
    eligible = screen.loc[
        (screen["abs_screen_correlation"] >= min_correlation)
        & (~screen["destination"].isin(ACTIVE_CONTROLS))
        & (screen["source_index"] != screen["destination_index"])
    ].copy()
    selected = (
        eligible.sort_values(
            ["destination", "abs_screen_correlation", "lag_steps", "source"],
            ascending=[True, False, True, True],
        )
        .groupby("destination", sort=False, as_index=False)
        .head(per_destination)
    )
    data_candidates: set[Triple] = {
        (int(row.source_index), int(row.destination_index), int(row.lag_steps))
        for row in selected.itertuples(index=False)
    }
    # Every non-control destination also receives its own history as an
    # autoregressive conditioning candidate. Otherwise highly persistent sensor
    # series can create spurious cross-sensor links simply because Y(t-1) was not
    # available to PC1/MCI. These candidates do not change the top-30 cross-edge
    # budget and are reported separately from the physical prior.
    autoregressive_candidates: set[Triple] = {
        (index, index, lag)
        for index, name in enumerate(raw_columns)
        if name not in ACTIVE_CONTROLS
        for lag in sorted(screen["lag_steps"].unique())
    }
    union = data_candidates | physical | autoregressive_candidates
    screen_lookup = {
        (int(row.source_index), int(row.destination_index), int(row.lag_steps)): row
        for row in screen.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for triple in sorted(union):
        source, destination, lag = triple
        screen_row = screen_lookup[triple]
        in_data = triple in data_candidates
        in_physical = triple in physical
        in_autoregressive = triple in autoregressive_candidates
        meta = physical_metadata.get(triple, {})
        sources = []
        if in_data:
            sources.append("data")
        if in_physical:
            sources.append("physical")
        if in_autoregressive:
            sources.append("autoregressive")
        rows.append(
            {
                "source": raw_columns[source],
                "destination": raw_columns[destination],
                "source_index": source,
                "destination_index": destination,
                "lag_steps": lag,
                "lag_seconds": lag * SAMPLE_PERIOD_SECONDS,
                "screen_correlation": float(screen_row.screen_correlation),
                "abs_screen_correlation": float(screen_row.abs_screen_correlation),
                "data_screened": in_data,
                "physical_candidate": in_physical,
                "autoregressive_candidate": in_autoregressive,
                "candidate_source": "+".join(sources),
                "physical_relation_type": meta.get("physical_relation_type", ""),
                "physical_mechanism": meta.get("physical_mechanism", ""),
            }
        )
    return data_candidates, union, pd.DataFrame(rows)


def link_assumptions(
    triples: set[Triple], variable_count: int
) -> dict[int, dict[tuple[int, int], str]]:
    assumptions: dict[int, dict[tuple[int, int], str]] = {
        index: {} for index in range(variable_count)
    }
    for source, destination, lag in triples:
        assumptions[destination][(source, -lag)] = "-?>"
    return assumptions


def make_pcmci(
    segments: dict[int, np.ndarray], raw_columns: tuple[str, ...], verbosity: int
):
    try:
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI
    except ImportError as exc:
        raise RuntimeError("Install code/requirements.txt in the pytorch environment") from exc
    data = pp.DataFrame(
        data={key: value for key, value in segments.items()},
        analysis_mode="multiple",
        var_names=list(raw_columns),
    )
    return PCMCI(
        dataframe=data,
        cond_ind_test=ParCorr(significance="analytic"),
        verbosity=verbosity,
    )


def run_screened_pcmci(
    segments: dict[int, np.ndarray],
    raw_columns: tuple[str, ...],
    triples: set[Triple],
    max_lag: int,
    pc_alpha: float,
    max_conds_dim: int,
    verbosity: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, list[tuple[int, int]]]]:
    pcmci = make_pcmci(segments, raw_columns, verbosity)
    assumptions = link_assumptions(triples, len(raw_columns))
    parents = pcmci.run_pc_stable(
        link_assumptions=assumptions,
        tau_min=1,
        tau_max=max_lag,
        pc_alpha=pc_alpha,
        max_conds_dim=max_conds_dim,
        max_combinations=1,
    )
    result = pcmci.run_mci(
        link_assumptions=assumptions,
        tau_min=1,
        tau_max=max_lag,
        parents=parents,
        max_conds_py=max_conds_dim,
        max_conds_px=max_conds_dim,
        alpha_level=0.05,
        fdr_method="none",
    )
    return np.asarray(result["val_matrix"]), np.asarray(result["p_matrix"]), parents


def run_fixed_mci(
    segments: dict[int, np.ndarray],
    raw_columns: tuple[str, ...],
    triples: set[Triple],
    parents: dict[int, list[tuple[int, int]]],
    max_lag: int,
    max_conds_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    pcmci = make_pcmci(segments, raw_columns, 0)
    result = pcmci.run_mci(
        link_assumptions=link_assumptions(triples, len(raw_columns)),
        tau_min=1,
        tau_max=max_lag,
        parents=parents,
        max_conds_py=max_conds_dim,
        max_conds_px=max_conds_dim,
        alpha_level=0.05,
        fdr_method="none",
    )
    return np.asarray(result["val_matrix"]), np.asarray(result["p_matrix"])


def stationarity_table(
    records: list[dict[str, Any]], raw_columns: tuple[str, ...]
) -> pd.DataFrame:
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return pd.DataFrame({"variable": raw_columns, "note": "statsmodels unavailable"})
    arrays = sorted(
        (np.asarray(record["values"], dtype=np.float64) for record in records),
        key=len,
        reverse=True,
    )[:5]
    rows: list[dict[str, Any]] = []
    for column, variable in enumerate(raw_columns):
        level_p: list[float] = []
        difference_p: list[float] = []
        for values in arrays:
            series = values[:, column]
            if len(series) > 2000:
                center = (len(series) - 2000) // 2
                series = series[center : center + 2000]
            if len(series) < 40 or np.std(series) < 1e-10:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    level_p.append(float(adfuller(series, autolag="AIC")[1]))
                    difference = np.diff(series)
                    if np.std(difference) >= 1e-10:
                        difference_p.append(float(adfuller(difference, autolag="AIC")[1]))
            except (ValueError, np.linalg.LinAlgError):
                continue
        rows.append(
            {
                "variable": variable,
                "segments_tested_level": len(level_p),
                "median_adf_p_level": float(np.median(level_p)) if level_p else np.nan,
                "segments_tested_difference": len(difference_p),
                "median_adf_p_difference": (
                    float(np.median(difference_p)) if difference_p else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def serializable_parent_counts(
    parents: dict[int, list[tuple[int, int]]], raw_columns: tuple[str, ...]
) -> dict[str, int]:
    return {raw_columns[index]: len(value) for index, value in parents.items()}


def main() -> None:
    args = parse_args()
    validate_args(args)
    data_dir = args.data_dir.resolve()
    train_file = (args.train_file or data_dir / "train_clean.pkl").resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_columns = load_raw_columns(data_dir)
    records, data_audit = load_original_training_records(
        train_file, raw_columns, min_length=args.max_lag + 40
    )
    if not args.skip_stationarity:
        stationarity_table(records, raw_columns).to_csv(
            output_dir / "stationarity_audit.csv", index=False
        )

    screen_records, inference_records, screen_parents, inference_parents = split_parent_runs(
        records, args.seed
    )
    screen_segments, _, screen_scaler = transform_records(
        screen_records, raw_columns, args.transform, args.max_samples_per_segment
    )
    inference_segments, inference_parent_map, inference_scaler = transform_records(
        inference_records, raw_columns, args.transform, args.max_samples_per_segment
    )

    screen = lagged_correlation_screen(screen_segments, raw_columns, args.max_lag)
    screen.to_csv(output_dir / "lagged_correlation_screen.csv", index=False)
    physical, physical_metadata = expand_static_candidates(raw_columns, args.max_lag)
    data_candidates, candidates, candidate_inventory = select_candidates(
        screen,
        raw_columns,
        physical,
        physical_metadata,
        args.screen_per_destination,
        args.screen_min_correlation,
    )
    candidate_inventory.to_csv(output_dir / "pcmci_candidate_inventory.csv", index=False)
    flag_by_triple = {
        (int(row.source_index), int(row.destination_index), int(row.lag_steps)): row
        for row in candidate_inventory.itertuples(index=False)
    }
    print(
        f"transform={args.transform} correlations={len(screen)} "
        f"data_candidates={len(data_candidates)} physical_candidates={len(physical)} "
        f"PCMCI_union={len(candidates)} inference_segments={len(inference_segments)}",
        flush=True,
    )

    value, p_value, parents = run_screened_pcmci(
        inference_segments,
        raw_columns,
        candidates,
        args.max_lag,
        args.pc_alpha,
        args.max_conds_dim,
        args.verbosity,
    )
    ordered = sorted(candidates)
    q_values = fdr_bh(np.asarray([float(p_value[triple]) for triple in ordered]))
    tested_rows: list[dict[str, Any]] = []
    for triple, q_value in zip(ordered, q_values):
        source, destination, lag = triple
        flag = flag_by_triple[triple]
        effect = float(value[triple])
        tested_rows.append(
            {
                "source": raw_columns[source],
                "destination": raw_columns[destination],
                "source_index": source,
                "destination_index": destination,
                "lag_steps": lag,
                "lag_seconds": lag * SAMPLE_PERIOD_SECONDS,
                "source_representation": args.transform,
                "effect": effect,
                "abs_effect": abs(effect),
                "p_value": float(p_value[triple]),
                "q_value": float(q_value),
                "data_screened": bool(flag.data_screened),
                "physical_candidate": bool(flag.physical_candidate),
                "autoregressive_candidate": bool(flag.autoregressive_candidate),
                "candidate_source": flag.candidate_source,
                "physical_relation_type": flag.physical_relation_type,
                "physical_mechanism": flag.physical_mechanism,
            }
        )
    tests = pd.DataFrame(tested_rows)
    bootstrap_pool = tests.loc[
        (tests["q_value"] <= args.bootstrap_candidate_q)
        & (tests["abs_effect"] >= args.min_effect)
    ].copy()
    bootstrap_pool_before_limit = (
        bootstrap_pool.sort_values(
            ["source", "destination", "abs_effect"], ascending=[True, True, False]
        )
        .groupby(["source", "destination"], sort=False, as_index=False)
        .head(args.top_lags_per_pair)
        .sort_values("abs_effect", ascending=False)
    )
    bootstrap_pool = bootstrap_pool_before_limit
    if (
        args.max_bootstrap_candidates > 0
        and len(bootstrap_pool_before_limit) > args.max_bootstrap_candidates
    ):
        # A user-requested compute cap must never silently exclude a qualified
        # physical/H_mix candidate. Physical candidates are kept first; the
        # remaining slots are filled by strongest data-only candidates.
        forced = bootstrap_pool_before_limit.loc[
            bootstrap_pool_before_limit["physical_candidate"]
        ]
        optional = bootstrap_pool_before_limit.loc[
            ~bootstrap_pool_before_limit["physical_candidate"]
        ]
        remaining = max(0, args.max_bootstrap_candidates - len(forced))
        bootstrap_pool = pd.concat(
            [forced, optional.head(remaining)], ignore_index=True
        ).drop_duplicates(["source", "destination", "lag_steps"])
    bootstrap_triples: set[Triple] = {
        (int(row.source_index), int(row.destination_index), int(row.lag_steps))
        for row in bootstrap_pool.itertuples(index=False)
    }
    hit_count = {triple: 0 for triple in bootstrap_triples}
    effect_samples: dict[Triple, list[float]] = {triple: [] for triple in bootstrap_triples}
    rng = np.random.default_rng(args.seed + 1)
    for iteration in range(args.bootstrap):
        if not bootstrap_triples:
            break
        boot_segments = bootstrap_parent_runs(inference_segments, inference_parent_map, rng)
        boot_value, boot_p = run_fixed_mci(
            boot_segments,
            raw_columns,
            bootstrap_triples,
            parents,
            args.max_lag,
            args.max_conds_dim,
        )
        boot_order = sorted(bootstrap_triples)
        boot_q = fdr_bh(np.asarray([float(boot_p[triple]) for triple in boot_order]))
        for triple, q_value in zip(boot_order, boot_q):
            current_effect = float(boot_value[triple])
            effect_samples[triple].append(current_effect)
            if q_value <= args.fdr_alpha and np.sign(current_effect) == np.sign(float(value[triple])):
                hit_count[triple] += 1
        print(f"parent-run bootstrap {iteration + 1}/{args.bootstrap}", flush=True)

    effective_bootstrap = args.bootstrap if bootstrap_triples else 0
    stability = {
        triple: hit_count[triple] / effective_bootstrap if effective_bootstrap else 1.0
        for triple in bootstrap_triples
    }
    tests["bootstrap_stability"] = [
        stability.get(
            (int(row.source_index), int(row.destination_index), int(row.lag_steps)), np.nan
        )
        for row in tests.itertuples(index=False)
    ]
    tests["bootstrap_effect_sd"] = [
        (
            float(np.std(samples, ddof=1)) if len(samples) > 1 else np.nan
        )
        for samples in (
            effect_samples.get(
                (int(row.source_index), int(row.destination_index), int(row.lag_steps)), []
            )
            for row in tests.itertuples(index=False)
        )
    ]
    tests.to_csv(output_dir / "pcmci_candidate_tests.csv", index=False)

    final_all = tests.loc[
        (tests["q_value"] <= args.fdr_alpha)
        & (tests["abs_effect"] >= args.min_effect)
        & (tests["bootstrap_stability"] >= args.min_stability)
    ].copy()
    final_all["selection_score"] = final_all["abs_effect"] * np.sqrt(
        final_all["bootstrap_stability"]
    )
    final_all = final_all.sort_values("selection_score", ascending=False)
    final_all.to_csv(output_dir / "stable_pcmci_edges_all.csv", index=False)
    final_export = final_all.head(args.max_export_edges).copy()
    final_export.to_csv(output_dir / "stable_pcmci_edges.csv", index=False)
    stable_physical = final_all.loc[final_all["physical_candidate"]].copy()
    stable_physical.to_csv(output_dir / "stable_physical_lags.csv", index=False)
    proxy_tests = tests.loc[
        tests["physical_relation_type"].eq("latent_H_mix_proxy")
    ].copy()
    proxy_tests.to_csv(output_dir / "latent_H_mix_candidate_tests.csv", index=False)
    proxy = final_all.loc[
        final_all["physical_relation_type"].eq("latent_H_mix_proxy")
    ].copy()
    proxy.to_csv(output_dir / "latent_H_mix_proxy_lags.csv", index=False)
    unsupported_static = tests.loc[
        tests["physical_candidate"]
        & ~(
            (tests["q_value"] <= args.fdr_alpha)
            & (tests["abs_effect"] >= args.min_effect)
            & (tests["bootstrap_stability"] >= args.min_stability)
        )
    ].copy()
    unsupported_static.to_csv(output_dir / "unsupported_static_candidates.csv", index=False)

    def build_edge_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for row in frame.to_dict("records"):
            result.append(
                {
                    "source": row["source"],
                    "destination": row["destination"],
                    "lag_steps": int(row["lag_steps"]),
                    "lag_seconds": int(row["lag_seconds"]),
                    "source_representation": row["source_representation"],
                    "effect": float(row["effect"]),
                    "q_value": float(row["q_value"]),
                    "bootstrap_stability": float(row["bootstrap_stability"]),
                    "candidate_source": row["candidate_source"],
                    "physical_relation_type": row["physical_relation_type"],
                }
            )
        return result

    export_edge_records = build_edge_records(final_export)
    graph_payload = {
        "schema": "raw54_lagged_pcmci_v1",
        "transform": args.transform,
        "sample_period_seconds": SAMPLE_PERIOD_SECONDS,
        "nodes": list(raw_columns),
        "active_controls_exogenous": sorted(ACTIVE_CONTROLS),
        "edges": export_edge_records,
        "gnn_read_semantics": (
            "For each edge X(t-lag)->Y(t), read delta X at that lag when transform=difference; "
            "read X level at that lag when transform=level. H_mix proxy edges parameterize the "
            "latent KAN merge block and are not direct final physical edges."
        ),
    }
    (output_dir / "stable_lagged_graph.json").write_text(
        json.dumps(graph_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    graph_all_payload = dict(graph_payload)
    graph_all_payload["schema"] = "raw54_lagged_pcmci_all_stable_v2"
    graph_all_payload["edges"] = build_edge_records(final_all)
    (output_dir / "stable_lagged_graph_all.json").write_text(
        json.dumps(graph_all_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "scalers.json").write_text(
        json.dumps(
            {"raw_columns": raw_columns, "screen": screen_scaler, "inference": inference_scaler},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    audit = {
        "method": "disjoint parent-run marginal screen + restricted PC1/MCI + parent-run fixed-MCI bootstrap",
        "python": sys.version,
        "platform": platform.platform(),
        "training_file": str(train_file),
        **data_audit,
        "included_0617": "0617-ALL" in {record["parent_id"] for record in records},
        "validation_loaded": False,
        "test_loaded": False,
        "engineered_features_used": False,
        "raw_variables": len(raw_columns),
        "variables": list(raw_columns),
        "active_controls_treated_as_exogenous": sorted(ACTIVE_CONTROLS),
        "transform": args.transform,
        "transform_interpretation": (
            "X level predicts Y level" if args.transform == "level"
            else "change in X predicts change in Y"
        ),
        "max_lag_steps": args.max_lag,
        "max_lag_seconds": args.max_lag * SAMPLE_PERIOD_SECONDS,
        "all_marginal_correlations_computed": len(screen),
        "correlation_formula": "number_of_variables^2 * max_lag_steps",
        "screen_parent_runs": screen_parents,
        "inference_parent_runs": inference_parents,
        "screen_inference_parent_runs_disjoint": not bool(set(screen_parents) & set(inference_parents)),
        "screen_per_noncontrol_destination": args.screen_per_destination,
        "screen_min_abs_correlation": args.screen_min_correlation,
        "data_screen_candidates": len(data_candidates),
        "physical_and_H_mix_proxy_candidates": len(physical),
        "autoregressive_conditioning_candidates": int(
            candidate_inventory["autoregressive_candidate"].sum()
        ),
        "pcmci_union_candidates": len(candidates),
        "pc_alpha": args.pc_alpha,
        "max_conds_dim": args.max_conds_dim,
        "fdr_alpha": args.fdr_alpha,
        "bootstrap_candidate_q": args.bootstrap_candidate_q,
        "bootstrap_runs_requested": args.bootstrap,
        "bootstrap_type": "whole parent-run resampling, fixed full-fit parents, q-sign stable",
        "bootstrap_candidates_before_compute_limit": len(bootstrap_pool_before_limit),
        "max_bootstrap_candidates": args.max_bootstrap_candidates,
        "bootstrap_candidates": len(bootstrap_triples),
        "qualified_candidates_not_bootstrapped": int(
            len(bootstrap_pool_before_limit) - len(bootstrap_pool)
        ),
        "min_stability": args.min_stability,
        "min_effect": args.min_effect,
        "top_lags_per_pair": args.top_lags_per_pair,
        "max_export_edges": args.max_export_edges,
        "stable_edges_all": len(final_all),
        "stable_edges_exported": len(final_export),
        "stable_H_mix_proxy_edges": len(proxy),
        "pc1_parent_counts": serializable_parent_counts(parents, raw_columns),
    }
    (output_dir / "discovery_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
