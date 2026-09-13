#!/usr/bin/env python3
"""Efficient, leakage-safe PCMCI discovery with fixed-structure MCI bootstrap.

The expensive PC1 search is run once on a screened candidate graph. Bootstrap
replicates re-estimate MCI values only for the preselected links while keeping
the full-fit conditioning parents fixed. Candidate screening and inference use
disjoint parent runs, so p/q values are not computed on the same runs that
selected the data-driven candidate links.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_pipeline import load_frame, original_training_segment_records
from graph_tools import create_graph_variants, read_json
from protocol import MAX_CAUSAL_LAG, RAW_COLUMNS, project_dir


Triple = tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    root = project_dir()
    parser = argparse.ArgumentParser(description="Screened PCMCI with efficient MCI bootstrap.")
    parser.add_argument("--train-file", type=Path, default=root / "processed_data" / "train_clean.pkl")
    parser.add_argument("--static-graph", type=Path, default=root / "graphs" / "static_physical_graph.json")
    parser.add_argument("--output-dir", type=Path, default=root / "graphs" / "generated")
    parser.add_argument("--max-lag", type=int, default=MAX_CAUSAL_LAG)
    parser.add_argument("--pc-alpha", type=float, default=0.05)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap-candidate-q", type=float, default=0.10)
    parser.add_argument("--bootstrap", type=int, default=20)
    parser.add_argument("--min-stability", type=float, default=0.60)
    parser.add_argument("--min-effect", type=float, default=0.03)
    parser.add_argument("--screen-per-destination", type=int, default=30)
    parser.add_argument("--screen-min-correlation", type=float, default=0.02)
    parser.add_argument("--top-lags-per-pair", type=int, default=3)
    parser.add_argument("--max-total-edges", type=int, default=200)
    parser.add_argument("--max-conds-dim", type=int, default=10)
    parser.add_argument("--transform", choices=("difference", "level"), default="difference")
    parser.add_argument("--max-samples-per-segment", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--verbosity", type=int, default=0)
    return parser.parse_args()


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


def transform_segments(
    segments: dict[int, np.ndarray], mode: str, max_samples: int
) -> dict[int, np.ndarray]:
    transformed: dict[int, np.ndarray] = {}
    valve_indices = [
        RAW_COLUMNS.index(name)
        for name in (
            "CV8312", "CV8311", "CV8310", "CV8313", "CV8300", "CV8351",
            "EC-V2", "COOLDOWN",
        )
    ]
    for key, values in segments.items():
        array = np.diff(values, axis=0) if mode == "difference" else values.copy()
        if max_samples > 0 and len(array) > max_samples:
            # Preserve the 10 s interval and retain the most valve-active window.
            if mode == "difference":
                activity = np.abs(array[:, valve_indices]).sum(axis=1)
            else:
                activity = np.abs(
                    np.diff(
                        array[:, valve_indices], axis=0,
                        prepend=array[:1, valve_indices],
                    )
                ).sum(axis=1)
            cumulative = np.r_[0.0, np.cumsum(activity)]
            scores = cumulative[max_samples:] - cumulative[:-max_samples]
            start = int(np.argmax(scores))
            array = array[start : start + max_samples]
        transformed[key] = array
    stacked = np.concatenate(list(transformed.values()), axis=0)
    mean = stacked.mean(axis=0)
    scale = stacked.std(axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return {key: (array - mean) / scale for key, array in transformed.items()}


def split_parent_runs(
    records: list[dict[str, object]], seed: int
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str], list[str]]:
    groups = sorted({str(record["source_group"]) for record in records})
    if len(groups) < 4:
        raise ValueError("At least four original parent runs are required for independent screening/inference")
    rng = np.random.default_rng(seed)
    shuffled = [str(value) for value in np.asarray(groups, dtype=object)[rng.permutation(len(groups))]]
    screen_groups = set(shuffled[::2])
    inference_groups = set(groups) - screen_groups
    screen = [record for record in records if str(record["source_group"]) in screen_groups]
    inference = [record for record in records if str(record["source_group"]) in inference_groups]
    if not screen or not inference:
        raise AssertionError("Parent-run split is empty")
    return screen, inference, sorted(screen_groups), sorted(inference_groups)


def records_to_segments(
    records: list[dict[str, object]], mode: str, max_samples: int
) -> dict[int, np.ndarray]:
    arrays = {
        index: np.asarray(record["values"], dtype=np.float64)
        for index, record in enumerate(records)
    }
    return transform_segments(arrays, mode, max_samples)


def lagged_correlation_screen(segments: dict[int, np.ndarray], max_lag: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    variable_count = len(RAW_COLUMNS)
    for lag in range(1, max_lag + 1):
        source_values = np.concatenate(
            [values[:-lag] for values in segments.values() if len(values) > lag], axis=0
        )
        destination_values = np.concatenate(
            [values[lag:] for values in segments.values() if len(values) > lag], axis=0
        )
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
        for source in range(variable_count):
            for destination in range(variable_count):
                rows.append(
                    {
                        "source": RAW_COLUMNS[source], "destination": RAW_COLUMNS[destination],
                        "source_index": source, "destination_index": destination,
                        "lag_steps": lag, "lag_seconds": lag * 10,
                        "screen_correlation": float(correlation[source, destination]),
                        "abs_screen_correlation": float(abs(correlation[source, destination])),
                    }
                )
    return pd.DataFrame(rows)


def physical_candidate_triples(static_path: Path, max_lag: int) -> set[Triple]:
    index = {name: position for position, name in enumerate(RAW_COLUMNS)}
    triples: set[Triple] = set()
    for relation in read_json(static_path)["relations"]:
        maximum = min(int(relation["lag_max"]), max_lag)
        for lag in range(int(relation["lag_min"]), maximum + 1):
            triples.add(
                (index[str(relation["source"])], index[str(relation["destination"])], lag)
            )
    return triples


def select_candidate_triples(
    screen: pd.DataFrame,
    physical: set[Triple],
    per_destination: int,
    min_correlation: float,
) -> tuple[set[Triple], set[Triple], pd.DataFrame]:
    eligible = screen.loc[screen["abs_screen_correlation"] >= min_correlation].copy()
    selected = (
        eligible.sort_values(["destination", "abs_screen_correlation"], ascending=[True, False])
        .groupby("destination", sort=False, as_index=False)
        .head(per_destination)
    )
    data_triples = {
        (int(row.source_index), int(row.destination_index), int(row.lag_steps))
        for row in selected.itertuples(index=False)
    }
    union = data_triples | physical
    flags = []
    for triple in sorted(union):
        flags.append(
            {
                "source": RAW_COLUMNS[triple[0]], "destination": RAW_COLUMNS[triple[1]],
                "source_index": triple[0], "destination_index": triple[1],
                "lag_steps": triple[2], "lag_seconds": triple[2] * 10,
                "data_screened": triple in data_triples,
                "physical_candidate": triple in physical,
                "candidate_source": (
                    "data+physical" if triple in data_triples and triple in physical
                    else "data" if triple in data_triples else "physical"
                ),
            }
        )
    return data_triples, union, pd.DataFrame(flags)


def link_assumptions(triples: set[Triple]) -> dict[int, dict[tuple[int, int], str]]:
    assumptions: dict[int, dict[tuple[int, int], str]] = {
        index: {} for index in range(len(RAW_COLUMNS))
    }
    for source, destination, lag in triples:
        assumptions[destination][(source, -lag)] = "-?>"
    return assumptions


def make_pcmci(segments: dict[int, np.ndarray], verbosity: int):
    try:
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt in the active training environment") from exc
    data = pp.DataFrame(
        data={key: value for key, value in segments.items()},
        analysis_mode="multiple", var_names=list(RAW_COLUMNS),
    )
    return PCMCI(
        dataframe=data, cond_ind_test=ParCorr(significance="analytic"), verbosity=verbosity
    )


def run_screened_pcmci(
    segments: dict[int, np.ndarray],
    triples: set[Triple],
    max_lag: int,
    pc_alpha: float,
    max_conds_dim: int,
    verbosity: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, list[tuple[int, int]]]]:
    pcmci = make_pcmci(segments, verbosity)
    assumptions = link_assumptions(triples)
    parents = pcmci.run_pc_stable(
        link_assumptions=assumptions, tau_min=1, tau_max=max_lag,
        pc_alpha=pc_alpha, max_conds_dim=max_conds_dim, max_combinations=1,
    )
    result = pcmci.run_mci(
        link_assumptions=assumptions, tau_min=1, tau_max=max_lag,
        parents=parents, max_conds_py=max_conds_dim, max_conds_px=max_conds_dim,
        alpha_level=0.05, fdr_method="none",
    )
    return np.asarray(result["val_matrix"]), np.asarray(result["p_matrix"]), parents


def run_fixed_mci(
    segments: dict[int, np.ndarray],
    triples: set[Triple],
    parents: dict[int, list[tuple[int, int]]],
    max_lag: int,
    max_conds_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    pcmci = make_pcmci(segments, 0)
    result = pcmci.run_mci(
        link_assumptions=link_assumptions(triples), tau_min=1, tau_max=max_lag,
        parents=parents, max_conds_py=max_conds_dim, max_conds_px=max_conds_dim,
        alpha_level=0.05, fdr_method="none",
    )
    return np.asarray(result["val_matrix"]), np.asarray(result["p_matrix"])


def stationarity_table(records: list[dict[str, object]]) -> pd.DataFrame:
    try:
        from statsmodels.tools.sm_exceptions import SingularMatrixWarning
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return pd.DataFrame({"variable": RAW_COLUMNS, "note": "statsmodels unavailable"})
    rows = []
    arrays = sorted(
        (np.asarray(record["values"]) for record in records), key=len, reverse=True
    )[:5]
    for column, name in enumerate(RAW_COLUMNS):
        level_p: list[float] = []
        difference_p: list[float] = []
        for values in arrays:
            series = values[:, column]
            if len(series) > 2000:
                start = (len(series) - 2000) // 2
                series = series[start : start + 2000]
            if len(series) < 40 or np.std(series) < 1e-10:
                continue
            try:
                with warnings.catch_warnings():
                    # Piecewise-constant valve signals can make the optional ADF
                    # audit singular. This audit does not drive graph selection.
                    warnings.simplefilter("ignore", FutureWarning)
                    warnings.simplefilter("ignore", RuntimeWarning)
                    warnings.simplefilter("ignore", SingularMatrixWarning)
                    level_p.append(
                        float(adfuller(series, autolag="AIC", result_object=False)[1])
                    )
                    difference = np.diff(series)
                    if np.std(difference) >= 1e-10:
                        difference_p.append(
                            float(
                                adfuller(
                                    difference, autolag="AIC", result_object=False
                                )[1]
                            )
                        )
            except (ValueError, np.linalg.LinAlgError):
                continue
        rows.append(
            {
                "variable": name, "segments_tested": len(level_p),
                "median_adf_p_level": float(np.median(level_p)) if level_p else np.nan,
                "median_adf_p_difference": float(np.median(difference_p)) if difference_p else np.nan,
            }
        )
    return pd.DataFrame(rows)


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.max_lag <= MAX_CAUSAL_LAG:
        raise ValueError(f"max-lag must be in [1, {MAX_CAUSAL_LAG}]")
    if not 1 <= args.top_lags_per_pair <= args.max_lag:
        raise ValueError("top-lags-per-pair is invalid")
    if args.max_total_edges < 1 or args.screen_per_destination < 1 or args.max_conds_dim < 1:
        raise ValueError("edge, screening and conditioning limits must be positive")
    if args.bootstrap < 0:
        raise ValueError("bootstrap must be non-negative")
    for name in ("pc_alpha", "fdr_alpha", "bootstrap_candidate_q", "min_stability"):
        if not 0.0 < float(getattr(args, name)) <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = original_training_segment_records(
        load_frame(args.train_file), min_length=args.max_lag + 40
    )
    stationarity_table(records).to_csv(args.output_dir / "stationarity_audit.csv", index=False)
    screen_records, inference_records, screen_groups, inference_groups = split_parent_runs(
        records, args.seed
    )
    screen_segments = records_to_segments(
        screen_records, args.transform, args.max_samples_per_segment
    )
    inference_segments = records_to_segments(
        inference_records, args.transform, args.max_samples_per_segment
    )

    screen_table = lagged_correlation_screen(screen_segments, args.max_lag)
    screen_table.to_csv(args.output_dir / "lagged_correlation_screen.csv", index=False)
    physical = physical_candidate_triples(args.static_graph, args.max_lag)
    data_candidates, candidates, candidate_table = select_candidate_triples(
        screen_table, physical, args.screen_per_destination, args.screen_min_correlation
    )
    candidate_table.to_csv(args.output_dir / "pcmci_candidate_inventory.csv", index=False)
    candidate_flags = {
        (int(row.source_index), int(row.destination_index), int(row.lag_steps)): row
        for row in candidate_table.itertuples(index=False)
    }

    print(
        f"screened candidates={len(data_candidates)} physical candidates={len(physical)} "
        f"union={len(candidates)} inference_segments={len(inference_segments)}",
        flush=True,
    )
    value, p_value, parents = run_screened_pcmci(
        inference_segments, candidates, args.max_lag, args.pc_alpha,
        args.max_conds_dim, args.verbosity,
    )
    ordered = sorted(candidates)
    q_values = fdr_bh(np.asarray([float(p_value[triple]) for triple in ordered]))
    main_rows = []
    for triple, q_value in zip(ordered, q_values):
        flag = candidate_flags[triple]
        effect = float(value[triple])
        main_rows.append(
            {
                "source": RAW_COLUMNS[triple[0]], "destination": RAW_COLUMNS[triple[1]],
                "source_index": triple[0], "destination_index": triple[1],
                "lag_steps": triple[2], "lag_seconds": triple[2] * 10,
                "effect": effect, "abs_effect": abs(effect),
                "p_value": float(p_value[triple]), "q_value": float(q_value),
                "data_screened": bool(flag.data_screened),
                "physical_candidate": bool(flag.physical_candidate),
                "candidate_source": flag.candidate_source,
            }
        )
    main_table = pd.DataFrame(main_rows)
    preselected = main_table.loc[
        (main_table["q_value"] <= args.bootstrap_candidate_q)
        & (main_table["abs_effect"] >= args.min_effect)
    ].copy()
    preselected = (
        preselected.sort_values(
            ["source", "destination", "abs_effect"], ascending=[True, True, False]
        )
        .groupby(["source", "destination"], sort=False, as_index=False)
        .head(args.top_lags_per_pair)
        .sort_values("abs_effect", ascending=False)
        .head(args.max_total_edges)
    )
    if preselected.empty:
        main_table.to_csv(args.output_dir / "pcmci_candidate_tests.csv", index=False)
        raise ValueError("No link survived the main-fit candidate thresholds")
    bootstrap_triples: set[Triple] = {
        (int(row.source_index), int(row.destination_index), int(row.lag_steps))
        for row in preselected.itertuples(index=False)
    }

    hit_count = {triple: 0 for triple in bootstrap_triples}
    effect_samples: dict[Triple, list[float]] = {
        triple: [] for triple in bootstrap_triples
    }
    rng = np.random.default_rng(args.seed + 1)
    inference_arrays = list(inference_segments.values())
    for iteration in range(args.bootstrap):
        sampled = rng.choice(
            len(inference_arrays), size=len(inference_arrays), replace=True
        )
        boot_segments = {
            index: inference_arrays[int(source)] for index, source in enumerate(sampled)
        }
        boot_value, boot_p = run_fixed_mci(
            boot_segments, bootstrap_triples, parents, args.max_lag, args.max_conds_dim
        )
        boot_order = sorted(bootstrap_triples)
        boot_q = fdr_bh(
            np.asarray([float(boot_p[triple]) for triple in boot_order])
        )
        for triple, q_value in zip(boot_order, boot_q):
            current_effect = float(boot_value[triple])
            effect_samples[triple].append(current_effect)
            main_sign = np.sign(float(value[triple]))
            if q_value <= args.fdr_alpha and np.sign(current_effect) == main_sign:
                hit_count[triple] += 1
        print(f"fixed-MCI bootstrap {iteration + 1}/{args.bootstrap}", flush=True)

    stability = {
        triple: hit_count[triple] / args.bootstrap if args.bootstrap else 1.0
        for triple in bootstrap_triples
    }
    table_stability = []
    table_effect_sd = []
    for row in main_table.itertuples(index=False):
        triple = (int(row.source_index), int(row.destination_index), int(row.lag_steps))
        samples = effect_samples.get(triple, [])
        table_stability.append(stability.get(triple, np.nan))
        table_effect_sd.append(float(np.std(samples, ddof=1)) if len(samples) > 1 else np.nan)
    main_table["bootstrap_stability"] = table_stability
    main_table["bootstrap_effect_sd"] = table_effect_sd
    main_table.to_csv(args.output_dir / "pcmci_candidate_tests.csv", index=False)

    final = main_table.loc[
        (main_table["q_value"] <= args.fdr_alpha)
        & (main_table["abs_effect"] >= args.min_effect)
        & (main_table["bootstrap_stability"] >= args.min_stability)
    ].copy()
    final["selection_score"] = final["abs_effect"] * np.sqrt(
        final["bootstrap_stability"]
    )
    final = final.sort_values("selection_score", ascending=False).head(
        args.max_total_edges
    )
    if final.empty:
        raise ValueError("No link survived q/effect/bootstrap stability thresholds")

    selected_edges: list[dict[str, Any]] = []
    for row in final.to_dict("records"):
        effect_sd = row["bootstrap_effect_sd"]
        selected_edges.append(
            {
                "source": row["source"], "destination": row["destination"],
                "lag_steps": int(row["lag_steps"]), "lag_seconds": int(row["lag_seconds"]),
                "effect": float(row["effect"]), "p_value": float(row["p_value"]),
                "q_value": float(row["q_value"]),
                "bootstrap_stability": float(row["bootstrap_stability"]),
                "bootstrap_effect_sd": float(effect_sd) if np.isfinite(effect_sd) else None,
                "confidence_score": float(row["bootstrap_stability"]),
                "confidence": "high" if row["bootstrap_stability"] >= 0.8 else "medium",
                "data_screened": bool(row["data_screened"]),
                "physical_candidate": bool(row["physical_candidate"]),
                "candidate_source": row["candidate_source"],
            }
        )
    counts = create_graph_variants(
        selected_edges, args.static_graph, args.output_dir, args.seed
    )
    parent_counts = {RAW_COLUMNS[key]: len(parent_list) for key, parent_list in parents.items()}
    audit = {
        "method": "independent parent-run correlation screening + restricted PC1/MCI + fixed-parent MCI bootstrap",
        "training_file": str(args.train_file), "original_segments": len(records),
        "screen_parent_runs": screen_groups, "inference_parent_runs": inference_groups,
        "screen_inference_parent_runs_disjoint": not bool(set(screen_groups) & set(inference_groups)),
        "augmented_segments_used": False, "excluded_0617": True,
        "validation_used": False, "test_used": False, "variables": list(RAW_COLUMNS),
        "transform": args.transform, "max_lag_steps": args.max_lag,
        "max_lag_seconds": args.max_lag * 10,
        "max_samples_per_segment": args.max_samples_per_segment,
        "data_screen_candidates": len(data_candidates), "physical_candidates": len(physical),
        "pcmci_union_candidates": len(candidates),
        "bootstrap_candidates": len(bootstrap_triples),
        "pc_alpha": args.pc_alpha, "max_conds_dim": args.max_conds_dim,
        "fdr_alpha": args.fdr_alpha,
        "bootstrap_candidate_q": args.bootstrap_candidate_q,
        "bootstrap_runs": args.bootstrap,
        "bootstrap_type": "fixed-parent MCI, segment resampling, sign-stable",
        "min_stability": args.min_stability, "min_effect": args.min_effect,
        "top_lags_per_pair": args.top_lags_per_pair,
        "max_total_edges": args.max_total_edges,
        "parent_counts": parent_counts, "graph_edge_counts": counts,
    }
    (args.output_dir / "discovery_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
