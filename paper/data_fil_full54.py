#!/usr/bin/env python3
"""Adapt the 0822 causal dataset builder to all 54 enabled raw signals."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REFERENCE_SCRIPT = Path(__file__).resolve().parent / "0822" / "data_fil.py"

ENABLED_REQUIRED_SIGNALS = (
    "TE8307",
    "TE8308",
    "TE8309",
    "TE8310",
    "TE8330",
    "TE8339",
    "TE8350",
    "TE8351",
    "TE8352",
    "A管",
    "B管",
    "C管",
    "D管",
    "E管",
    "F管",
    "冷屏上",
    "冷屏下",
    "冷屏中",
    "FT8351",
    "PT8307",
    "PT8308A",
    "PT8308B",
    "PT8309",
    "PT8310",
    "PT8330",
    "PT8350",
    "PT8351",
    "PT8352",
    "PT8390",
    "PT8391",
    "CV8312",
    "CV8311",
    "CV8310",
    "CV8313",
    "CV8300",
    "CV8358",
    "CV8359",
    "CV8350",
    "CV8351",
    "CV8330",
    "CV8339",
    "CV8338",
    "CV8308",
    "CV8352",
    "PT02",
    "CV8305",
    "TE8353",
    "FC-V1",
    "Thv",
    "Tef",
    "Tcd",
    "DTbr",
)

ALL_ENABLED_VALVES = (
    "CV8312",
    "CV8311",
    "CV8310",
    "CV8313",
    "CV8300",
    "CV8358",
    "CV8359",
    "CV8350",
    "CV8351",
    "CV8330",
    "CV8339",
    "CV8338",
    "CV8308",
    "CV8352",
    "CV8305",
    "FC-V1",
    "EC-V2",
    "COOLDOWN",
)


def load_reference_module():
    spec = importlib.util.spec_from_file_location("reference_data_fil", REFERENCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {REFERENCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_reference_module()
    module.REQUIRED_RAW_SIGNAL_COLS = ENABLED_REQUIRED_SIGNALS
    module.RAW_SIGNAL_COLS = (*ENABLED_REQUIRED_SIGNALS, *module.OPTIONAL_MODULE_VALVE_COLS)
    module.PRIMARY_VALVE_COLS = ALL_ENABLED_VALVES

    original_parent_parser = module.canonical_parent_id

    def canonical_parent_id(source_group: str) -> str:
        normalized = Path(source_group).stem.lower().replace("_", "-")
        if "260721" in normalized and "back" in normalized:
            return "260721-BACK"
        return original_parent_parser(source_group)

    module.canonical_parent_id = canonical_parent_id
    args = module.parse_args()
    module.parse_args = lambda: args
    module.main()

    if args.validate_only:
        return

    config_path = args.output_dir / "data_build_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "raw_signal_scope": "all 54 enabled nonconstant signals after operator-confirmed pruning",
            "enabled_raw_signal_count": 54,
            "primary_valve_columns": list(ALL_ENABLED_VALVES),
            "fc_v1_semantics": "confirmed 0-100 percent valve opening",
            "split_rationale": (
                "260428 is the held-out complete validation process because 260501 is "
                "absent; the newest BACK process 260721-BACK is the held-out test"
            ),
            "back_policy": (
                "0623-BACK is training only; 260721-BACK is the held-out test "
                "(Original only)"
            ),
            "pcmci_scope_note": (
                "PCMCI should be fitted on Original training parents only; augmented "
                "variants are intended for predictive-model training"
            ),
        }
    )
    config["dynamic_test"]["selection_signal"] = (
        "causal event mask from all 18 enabled valve-opening channels"
    )
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    catalog_path = args.output_dir / "feature_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["policy"] = (
        "all 54 enabled raw signals plus the 0822 causal engineered features are "
        "materialized; training selects a feature subset and one future target"
    )
    catalog["enabled_raw_signal_count"] = 54
    catalog["primary_valve_columns"] = list(ALL_ENABLED_VALVES)
    catalog["not_materialized"].pop("FC_V1", None)
    catalog["pcmci_scope_note"] = (
        "use Original training parents only for causal discovery to avoid treating "
        "augmented copies as independent experiments"
    )
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
