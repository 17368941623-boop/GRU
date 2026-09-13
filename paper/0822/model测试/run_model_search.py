#!/usr/bin/env python3
"""Run the model/hyperparameter development grid sequentially."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_PROGRAM = SCRIPT_DIR / "train_model_ablation.py"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "outputs"


@dataclass(frozen=True)
class SearchConfig:
    model: str
    config_id: str
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential train/validation-only model parameter search."
    )
    parser.add_argument("--lookback", type=int, required=True)
    parser.add_argument(
        "--common-origin-lookback",
        type=int,
        default=None,
        help="Defaults to --lookback; must be at least as large as lookback.",
    )
    parser.add_argument("--predict-steps", type=int, default=15)
    parser.add_argument("--seeds", default="42,62")
    parser.add_argument("--profile", choices=("quick", "full"), default="full")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Number of disjoint terminal shards used for the search.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index handled by this process.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_seeds(text: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item.strip()) for item in text.split(",") if item.strip()}))
    if not values:
        raise ValueError("At least one seed is required")
    return values


def make_config(
    model: str,
    *,
    history_hidden: int = 64,
    graph_hidden: int = 64,
    graph_sweeps: int = 1,
    kan_grid: int = 8,
    tcn_levels: int = 3,
) -> SearchConfig:
    pieces = [f"hh{history_hidden}"]
    if "gnn" in model:
        pieces.extend((f"gh{graph_hidden}", f"gs{graph_sweeps}"))
    if "kan" in model:
        pieces.append(f"kg{kan_grid}")
    if "tcn" in model:
        pieces.append(f"tl{tcn_levels}")
    return SearchConfig(
        model=model,
        config_id="_".join(pieces),
        history_hidden=history_hidden,
        graph_hidden=graph_hidden,
        graph_sweeps=graph_sweeps,
        kan_grid=kan_grid,
        tcn_levels=tcn_levels,
    )


def tcn_level_candidates(lookback: int) -> tuple[int, int]:
    # With kernel=3 and dilations 1,2,..., the receptive field is
    # 1 + 2*(2**levels - 1).  Test one level below full coverage and the first
    # level that covers the selected lookback.
    covering = math.ceil(math.log2(max((lookback - 1) / 2.0 + 1.0, 2.0)))
    covering = min(max(covering, 2), 7)
    return max(2, covering - 1), covering


def development_grid(profile: str, lookback: int = 20) -> list[SearchConfig]:
    configs = [make_config("gru_baseline")]
    lower_tcn_level, covering_tcn_level = tcn_level_candidates(lookback)
    if profile == "quick":
        graph_pairs = ((32, 1), (64, 1), (64, 2))
        kan_triples = ((32, 1, 5), (64, 1, 8), (64, 2, 8))
        tcn_pairs = ((32, lower_tcn_level), (64, covering_tcn_level))
        graph_mlp_tcn = (
            (32, 1, lower_tcn_level),
            (64, 2, covering_tcn_level),
        )
        graph_tcn = (
            (32, 1, 5, lower_tcn_level),
            (64, 2, 8, covering_tcn_level),
        )
    else:
        graph_pairs = tuple(
            (hidden, sweeps) for hidden in (32, 64) for sweeps in (1, 2)
        )
        kan_triples = tuple(
            (hidden, sweeps, grid)
            for hidden in (32, 64)
            for sweeps in (1, 2)
            for grid in (5, 8, 12)
        )
        tcn_pairs = tuple(
            (hidden, levels)
            for hidden in (32, 64)
            for levels in (lower_tcn_level, covering_tcn_level)
        )
        graph_mlp_tcn = tuple(
            (hidden, sweeps, levels)
            for hidden in (32, 64)
            for sweeps in (1, 2)
            for levels in (lower_tcn_level, covering_tcn_level)
        )
        graph_tcn = tuple(
            (hidden, sweeps, grid, covering_tcn_level)
            for hidden in (32, 64)
            for sweeps in (1, 2)
            for grid in (5, 8, 12)
        )

    for model in ("serial_mlp_gnn_gru", "parallel_gru_mlp_gnn"):
        configs.extend(
            make_config(model, graph_hidden=hidden, graph_sweeps=sweeps)
            for hidden, sweeps in graph_pairs
        )
    for model in ("serial_kan_gnn_gru", "parallel_gru_kan_gnn"):
        configs.extend(
            make_config(
                model,
                graph_hidden=hidden,
                graph_sweeps=sweeps,
                kan_grid=grid,
            )
            for hidden, sweeps, grid in kan_triples
        )
    configs.extend(
        make_config("tcn_baseline", history_hidden=hidden, tcn_levels=levels)
        for hidden, levels in tcn_pairs
    )
    configs.extend(
        make_config(
            "serial_mlp_gnn_tcn",
            history_hidden=hidden,
            graph_hidden=hidden,
            graph_sweeps=sweeps,
            tcn_levels=levels,
        )
        for hidden, sweeps, levels in graph_mlp_tcn
    )
    configs.extend(
        make_config(
            "serial_kan_gnn_tcn",
            history_hidden=hidden,
            graph_hidden=hidden,
            graph_sweeps=sweeps,
            kan_grid=grid,
            tcn_levels=levels,
        )
        for hidden, sweeps, grid, levels in graph_tcn
    )
    identities = {(config.model, config.config_id) for config in configs}
    if len(identities) != len(configs):
        raise AssertionError("Duplicate model search configuration")
    return configs


def command_for(
    args: argparse.Namespace,
    config: SearchConfig,
    seed: int,
) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_PROGRAM),
        "--model", config.model,
        "--config-id", config.config_id,
        "--lookback", str(args.lookback),
        "--common-origin-lookback", str(args.common_origin_lookback),
        "--predict-steps", str(args.predict_steps),
        "--seed", str(seed),
        "--results-dir", str(args.results_dir),
        "--history-hidden", str(config.history_hidden),
        "--graph-hidden", str(config.graph_hidden),
        "--graph-sweeps", str(config.graph_sweeps),
        "--edge-hidden", str(config.edge_hidden),
        "--kan-grid", str(config.kan_grid),
        "--tcn-levels", str(config.tcn_levels),
        "--tcn-kernel", str(config.tcn_kernel),
        "--control-hidden", str(config.control_hidden),
        "--fusion-hidden", str(config.fusion_hidden),
        "--dropout", str(config.dropout),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--patience", str(args.patience),
        "--num-workers", str(args.num_workers),
    ]
    if args.data_dir is not None:
        command.extend(("--data-dir", str(args.data_dir)))
    if args.overwrite:
        command.append("--overwrite")
    return command


def metrics_path(
    results_dir: Path,
    lookback: int,
    predict_steps: int,
    config: SearchConfig,
    seed: int,
) -> Path:
    return (
        results_dir
        / "development"
        / f"horizon_{predict_steps:02d}"
        / f"lookback_{lookback:02d}"
        / config.model
        / config.config_id
        / f"seed_{seed}"
        / "metrics.json"
    )


def write_manifest_atomic(
    csv_path: Path,
    json_path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Safely publish identical full manifests from concurrent shards."""
    csv_temp = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    json_temp = json_path.with_name(f".{json_path.name}.{os.getpid()}.tmp")
    try:
        with csv_temp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        json_temp.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(csv_temp, csv_path)
        os.replace(json_temp, json_path)
    finally:
        csv_temp.unlink(missing_ok=True)
        json_temp.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.common_origin_lookback is None:
        args.common_origin_lookback = args.lookback
    if args.common_origin_lookback < args.lookback:
        raise ValueError("common-origin-lookback cannot be smaller than lookback")
    if args.shard_count < 1:
        raise ValueError("shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must satisfy 0 <= index < shard-count")
    seeds = parse_seeds(args.seeds)
    configs = development_grid(args.profile, args.lookback)
    manifest_dir = (
        args.results_dir
        / "development"
        / f"horizon_{args.predict_steps:02d}"
        / f"lookback_{args.lookback:02d}"
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for config in configs:
        for seed in seeds:
            row = asdict(config)
            row.update(
                {
                    "seed": seed,
                    "profile": args.profile,
                    "lookback": args.lookback,
                    "common_origin_lookback": args.common_origin_lookback,
                    "predict_steps": args.predict_steps,
                }
            )
            manifest_rows.append(row)
    write_manifest_atomic(
        manifest_dir / "search_manifest.csv",
        manifest_dir / "search_manifest.json",
        manifest_rows,
    )

    all_jobs = [
        (config, seed)
        for config in configs
        for seed in seeds
    ]
    shard_jobs = [
        (global_index, config, seed)
        for global_index, (config, seed) in enumerate(all_jobs)
        if global_index % args.shard_count == args.shard_index
    ]
    for local_index, (global_index, config, seed) in enumerate(shard_jobs, start=1):
        destination = metrics_path(
            args.results_dir,
            args.lookback,
            args.predict_steps,
            config,
            seed,
        )
        if destination.exists() and not args.overwrite:
            print(
                f"[shard {args.shard_index + 1}/{args.shard_count} | "
                f"{local_index}/{len(shard_jobs)} | global {global_index + 1}/{len(all_jobs)}] "
                f"SKIP existing "
                f"{config.model}/{config.config_id}/seed_{seed}",
                flush=True,
            )
            continue
        print(
            f"[shard {args.shard_index + 1}/{args.shard_count} | "
            f"{local_index}/{len(shard_jobs)} | global {global_index + 1}/{len(all_jobs)}] "
            f"START "
            f"{config.model}/{config.config_id}/seed_{seed}",
            flush=True,
        )
        subprocess.run(command_for(args, config, seed), check=True)
        print(
            f"[shard {args.shard_index + 1}/{args.shard_count} | "
            f"{local_index}/{len(shard_jobs)} | global {global_index + 1}/{len(all_jobs)}] "
            f"COMPLETE "
            f"{config.model}/{config.config_id}/seed_{seed}",
            flush=True,
        )
    print(f"SEARCH_SHARD_INDEX={args.shard_index}")
    print(f"SEARCH_SHARD_COUNT={args.shard_count}")
    print(f"SEARCH_SHARD_RUNS_COMPLETE={len(shard_jobs)}")
    print(f"SEARCH_GLOBAL_RUNS_EXPECTED={len(all_jobs)}")
    print(f"SEARCH_MANIFEST={manifest_dir / 'search_manifest.csv'}")


if __name__ == "__main__":
    main()
