#!/usr/bin/env python3
"""Prespecified 10-600 s conditional lag audit for the H_mix block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_io import bootstrap_parent_runs, load_original_training_records, transform_records
from discover_pcmci import fdr_bh, run_fixed_mci, run_screened_pcmci
from protocol import SAMPLE_PERIOD_SECONDS, load_raw_columns


NAMES = ("TE8352", "TE8353", "A管", "EC-V2", "COOLDOWN", "FC-V1", "Thv")
TARGET_SOURCES = ("TE8352", "TE8353", "A管", "EC-V2", "COOLDOWN", "FC-V1")
TARGET = "Thv"


def main() -> None:
    code_dir = Path(__file__).resolve().parent
    experiment_dir = code_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=experiment_dir.parent / "processed_data")
    parser.add_argument("--output-dir", type=Path, default=experiment_dir / "output" / "hmix_targeted" / "difference")
    parser.add_argument("--transform", choices=("difference", "level"), default="difference")
    parser.add_argument("--max-lag", type=int, default=60)
    parser.add_argument("--bootstrap", type=int, default=50)
    parser.add_argument("--pc-alpha", type=float, default=0.05)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--min-effect", type=float, default=0.02)
    parser.add_argument("--min-stability", type=float, default=0.60)
    parser.add_argument("--max-conds-dim", type=int, default=10)
    parser.add_argument(
        "--max-samples-per-segment", type=int, default=0,
        help="0 keeps every Original point; use a positive cap only for memory limits.",
    )
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    if args.max_lag < 1 or args.bootstrap < 1:
        raise ValueError("max-lag and bootstrap must be positive")

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_columns = load_raw_columns(data_dir)
    records, data_audit = load_original_training_records(
        data_dir / "train_clean.pkl", raw_columns, min_length=args.max_lag + 40
    )
    indices = [raw_columns.index(name) for name in NAMES]
    subset_records = [
        {
            "parent_id": record["parent_id"],
            "file_id": record["file_id"],
            "values": np.asarray(record["values"])[:, indices],
        }
        for record in records
    ]
    segments, parent_map, _ = transform_records(
        subset_records, NAMES, args.transform, args.max_samples_per_segment
    )
    index = {name: position for position, name in enumerate(NAMES)}
    target_triples = {
        (index[source], index[TARGET], lag)
        for source in TARGET_SOURCES
        for lag in range(1, args.max_lag + 1)
    }
    conditioning_triples = set(target_triples)
    for name in ("TE8352", "TE8353", "A管", "Thv"):
        for lag in range(1, args.max_lag + 1):
            conditioning_triples.add((index[name], index[name], lag))
    for source, destination in (("TE8352", "TE8353"), ("TE8353", "A管")):
        for lag in range(1, min(30, args.max_lag) + 1):
            conditioning_triples.add((index[source], index[destination], lag))

    value, p_value, parents = run_screened_pcmci(
        segments,
        NAMES,
        conditioning_triples,
        args.max_lag,
        args.pc_alpha,
        args.max_conds_dim,
        0,
    )
    ordered = sorted(target_triples)
    q_values = fdr_bh(np.asarray([float(p_value[triple]) for triple in ordered]))
    rng = np.random.default_rng(args.seed + 11)
    hit_count = {triple: 0 for triple in ordered}
    effect_samples = {triple: [] for triple in ordered}
    for iteration in range(args.bootstrap):
        boot_segments = bootstrap_parent_runs(segments, parent_map, rng)
        boot_value, boot_p = run_fixed_mci(
            boot_segments,
            NAMES,
            target_triples,
            parents,
            args.max_lag,
            args.max_conds_dim,
        )
        boot_q = fdr_bh(np.asarray([float(boot_p[triple]) for triple in ordered]))
        for triple, current_q in zip(ordered, boot_q):
            current_effect = float(boot_value[triple])
            effect_samples[triple].append(current_effect)
            if current_q <= args.fdr_alpha and np.sign(current_effect) == np.sign(float(value[triple])):
                hit_count[triple] += 1
        print(f"H_mix {args.transform} parent-run bootstrap {iteration + 1}/{args.bootstrap}", flush=True)

    rows = []
    for triple, q_value in zip(ordered, q_values):
        source, destination, lag = triple
        samples = effect_samples[triple]
        effect = float(value[triple])
        rows.append(
            {
                "source": NAMES[source],
                "destination": NAMES[destination],
                "lag_steps": lag,
                "lag_seconds": lag * SAMPLE_PERIOD_SECONDS,
                "source_representation": args.transform,
                "effect": effect,
                "abs_effect": abs(effect),
                "p_value": float(p_value[triple]),
                "q_value": float(q_value),
                "bootstrap_stability": hit_count[triple] / args.bootstrap,
                "bootstrap_effect_sd": (
                    float(np.std(samples, ddof=1)) if len(samples) > 1 else np.nan
                ),
            }
        )
    tests = pd.DataFrame(rows)
    tests.to_csv(output_dir / "hmix_targeted_candidate_tests.csv", index=False)
    stable = tests.loc[
        (tests.q_value <= args.fdr_alpha)
        & (tests.abs_effect >= args.min_effect)
        & (tests.bootstrap_stability >= args.min_stability)
    ].copy()
    stable["selection_score"] = stable.abs_effect * np.sqrt(stable.bootstrap_stability)
    stable = stable.sort_values(["source", "selection_score"], ascending=[True, False])
    stable.to_csv(output_dir / "hmix_targeted_stable_lags.csv", index=False)
    best = (
        stable.groupby("source", as_index=False).head(5)
        if not stable.empty else stable.copy()
    )
    best.to_csv(output_dir / "hmix_targeted_top5_per_source.csv", index=False)
    audit = {
        "method": "prespecified H_mix source-to-Thv conditional MCI; no marginal pre-screen",
        "interpretation": (
            "This estimates conditional predictive delays, not isolated physical transport time. "
            "Compare with hmix_lag_diagnostics event and lead/lag outputs."
        ),
        "transform": args.transform,
        "variables": list(NAMES),
        "target_sources": list(TARGET_SOURCES),
        "max_lag_steps": args.max_lag,
        "max_lag_seconds": args.max_lag * SAMPLE_PERIOD_SECONDS,
        "parent_runs_used": sorted(set(parent_map.values())),
        "parent_runs_count": len(set(parent_map.values())),
        "segments": len(segments),
        "target_lags_tested_and_bootstrapped": len(target_triples),
        "bootstrap_runs": args.bootstrap,
        "fdr_family": f"{len(target_triples)} prespecified source-lag tests into Thv",
        "min_effect": args.min_effect,
        "min_stability": args.min_stability,
        "stable_lags": len(stable),
        "validation_loaded": False,
        "test_loaded": False,
        "engineered_features_used": False,
        "data_audit": data_audit,
    }
    (output_dir / "hmix_targeted_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
