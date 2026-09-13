#!/usr/bin/env python3
"""Run the 0822 augmentation pipeline on all enabled temperature sensors."""

from __future__ import annotations

import importlib.util
from pathlib import Path


REFERENCE_SCRIPT = Path(__file__).resolve().parent / "0822" / "create.py"


def load_reference_module():
    spec = importlib.util.spec_from_file_location("reference_create", REFERENCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {REFERENCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_reference_module()
    additional_temperatures = {
        "C管",
        "D管",
        "E管",
        "F管",
        "冷屏上",
        "冷屏中",
        "冷屏下",
    }

    def is_augmented_temperature_column(column: str) -> bool:
        if not str(column).strip().lower().endswith(" valuey"):
            return False
        signal = module.signal_name(column)
        if signal in module.EXCLUDED_TEMPERATURE_SIGNALS:
            return False
        upper = signal.upper()
        return (
            upper.startswith("TE")
            or upper in module.EXACT_TEMPERATURE_SIGNALS
            or signal in additional_temperatures
        )

    module.is_augmented_temperature_column = is_augmented_temperature_column
    module.main()


if __name__ == "__main__":
    main()
