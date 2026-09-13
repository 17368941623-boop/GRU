#!/usr/bin/env python3
"""Frozen protocol for the Raw54 PCMCI discovery package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SAMPLE_PERIOD_SECONDS = 10
DEFAULT_MAX_LAG = 30
EXPECTED_RAW_VARIABLES = 54

# These are operator commands. They may cause downstream process changes, but the
# process must not be interpreted as causing the operator's future command.
ACTIVE_CONTROLS = frozenset(
    {
        "CV8312", "CV8311", "CV8310", "CV8313", "CV8300", "CV8358",
        "CV8359", "CV8350", "CV8351", "CV8330", "CV8339", "CV8338",
        "CV8308", "CV8352", "CV8305", "FC-V1", "EC-V2", "COOLDOWN",
    }
)

# Observable relations frozen from static_physical_graph_v2_review 1.0.
# Each tuple is (source, destination, lag_min_steps, lag_max_steps, mechanism).
STATIC_RELATIONS: tuple[tuple[str, str, int, int, str], ...] = (
    ("CV8300", "TE8310", 1, 6, "G1 inlet valve to upstream temperature"),
    ("CV8300", "PT8310", 1, 6, "G1 inlet valve to upstream pressure"),
    ("CV8313", "TE8310", 1, 6, "G2 inlet valve to upstream temperature"),
    ("CV8313", "PT8310", 1, 6, "G2 inlet valve to upstream pressure"),
    ("CV8310", "FT8351", 1, 6, "branch allocation to mainline flow"),
    ("CV8310", "TE8351", 1, 12, "branch allocation to downstream temperature"),
    ("CV8310", "PT8351", 1, 8, "branch allocation to downstream pressure"),
    ("CV8351", "FT8351", 1, 6, "A-line valve to mainline flow"),
    ("CV8351", "TE8351", 1, 12, "A-line valve to downstream temperature"),
    ("CV8351", "PT8351", 1, 8, "A-line valve to downstream pressure"),
    ("TE8310", "TE8351", 1, 18, "three-branch KAN bridge temperature constraint"),
    ("PT8310", "PT8351", 1, 12, "three-branch KAN bridge pressure constraint"),
    ("PT8310", "FT8351", 1, 12, "three-branch KAN bridge flow constraint"),
    ("TE8351", "TE8352", 1, 12, "mainline temperature transport"),
    ("PT8351", "PT8352", 1, 8, "mainline pressure transport"),
    ("FT8351", "TE8352", 1, 12, "flow-dependent thermal transport"),
    ("FT8351", "PT8352", 1, 8, "flow-dependent pressure transport"),
    ("TE8352", "TE8353", 1, 12, "mainline temperature transport"),
    ("PT8352", "TE8353", 1, 12, "pressure-dependent thermal transport"),
    ("FT8351", "TE8353", 1, 18, "flow-dependent downstream thermal transport"),
    ("CV8312", "TE8330", 1, 12, "branch-1 valve to local temperature"),
    ("CV8312", "PT8330", 1, 8, "branch-1 valve to local pressure"),
    ("CV8330", "TE8330", 1, 12, "branch-1 local valve to temperature"),
    ("CV8330", "PT8330", 1, 8, "branch-1 local valve to pressure"),
    ("CV8311", "TE8350", 1, 12, "branch-2 valve to local temperature"),
    ("CV8311", "PT8350", 1, 8, "branch-2 valve to local pressure"),
    ("CV8350", "TE8350", 1, 12, "branch-2 local valve to temperature"),
    ("CV8350", "PT8350", 1, 8, "branch-2 local valve to pressure"),
    ("TE8353", "A管", 1, 12, "mainline outlet to A-line temperature"),
)

# State-memory candidates are part of the physical prior, not arbitrary data edges.
SELF_LAG_RANGES: tuple[tuple[str, int], ...] = (
    ("TE8310", 30), ("PT8310", 15), ("TE8351", 30), ("PT8351", 15),
    ("FT8351", 15), ("TE8352", 30), ("PT8352", 15), ("TE8353", 30),
    ("Thv", 30), ("TE8330", 30), ("PT8330", 15), ("TE8350", 30),
    ("PT8350", 15), ("A管", 30),
)

# H_mix is latent and therefore cannot be tested directly by PCMCI. These proxy
# tests estimate delays for the later H_mix KAN block without adding a fake sensor.
LATENT_PROXY_RELATIONS: tuple[tuple[str, str, int, int, str], ...] = (
    ("A管", "Thv", 1, 30, "H_mix proxy: inlet thermal state"),
    ("EC-V2", "Thv", 1, 30, "H_mix proxy: split valve"),
    ("COOLDOWN", "Thv", 1, 30, "H_mix proxy: split valve"),
    ("FC-V1", "Thv", 1, 30, "H_mix proxy: module flow valve"),
)


def load_raw_columns(data_dir: Path) -> tuple[str, ...]:
    catalog_path = data_dir / "feature_catalog.json"
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    columns = tuple(str(value) for value in payload["raw_signal_columns"])
    if len(columns) != EXPECTED_RAW_VARIABLES or len(set(columns)) != len(columns):
        raise ValueError(
            f"feature_catalog.json must contain {EXPECTED_RAW_VARIABLES} unique raw signals; "
            f"got {len(columns)}"
        )
    missing_controls = sorted(ACTIVE_CONTROLS - set(columns))
    if missing_controls:
        raise ValueError(f"Missing active controls in Raw54 catalog: {missing_controls}")
    return columns


def expand_static_candidates(
    raw_columns: tuple[str, ...], max_lag: int
) -> tuple[set[tuple[int, int, int]], dict[tuple[int, int, int], dict[str, Any]]]:
    index = {name: position for position, name in enumerate(raw_columns)}
    triples: set[tuple[int, int, int]] = set()
    metadata: dict[tuple[int, int, int], dict[str, Any]] = {}

    def add_relation(
        source: str, destination: str, lag_min: int, lag_max: int,
        mechanism: str, relation_type: str,
    ) -> None:
        if source not in index or destination not in index:
            raise ValueError(f"Static relation references missing Raw54 signal: {source}->{destination}")
        if destination in ACTIVE_CONTROLS and source != destination:
            raise ValueError(f"Forbidden incoming physical edge to active control: {source}->{destination}")
        for lag in range(max(1, lag_min), min(max_lag, lag_max) + 1):
            triple = (index[source], index[destination], lag)
            triples.add(triple)
            metadata[triple] = {
                "physical_relation_type": relation_type,
                "physical_mechanism": mechanism,
            }

    for relation in STATIC_RELATIONS:
        add_relation(*relation, relation_type="observed_static")
    for name, lag_max in SELF_LAG_RANGES:
        add_relation(name, name, 1, lag_max, "state memory", "self_memory")
    for relation in LATENT_PROXY_RELATIONS:
        add_relation(*relation, relation_type="latent_H_mix_proxy")
    return triples, metadata

