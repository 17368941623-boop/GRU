#!/usr/bin/env python3
"""Apply the validated 0822 causal filter to the new 88-signal exports.

This adapter reuses the original implementation and extends only signal-type
recognition for newly exported temperature, pressure, flow, and valve tags.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path


REFERENCE_FILTER = Path(__file__).resolve().parent / "0822" / "data_filter.py"


def load_reference_module():
    spec = importlib.util.spec_from_file_location("reference_data_filter", REFERENCE_FILTER)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reference filter: {REFERENCE_FILTER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_full88_signal_rules(module) -> None:
    pipe_temperatures = {"C管", "D管", "E管", "F管", "冷屏上", "冷屏中", "冷屏下"}
    explicit_valves = {"FC-V1", "EC-V2", "COOLDOWN", "COOLDOWM"}

    def signal_kind(column: str) -> str:
        name = module.signal_name(column)
        upper = name.upper().replace(" ", "")
        if str(column).strip().endswith(" Time") or "TIME" in upper:
            return "time"
        if upper.startswith("CV") or upper in explicit_valves:
            return "valve"
        if name in module.EXCLUDED_TEMPERATURES:
            return "excluded_temperature"
        if upper.startswith(("PT", "PDT")) or "VAC" in upper:
            return "pressure"
        if upper.startswith(("FT", "FR")):
            return "flow"
        if (
            upper.startswith("TE")
            or upper in {"THV", "DTBR", "TEF", "TCD"}
            or name in pipe_temperatures
            or re.fullmatch(r"(?:FC|EC)-[A-Z]+-T", upper)
            or re.fullmatch(r"(?:LA|LB|LG)-T\d+", upper)
            or re.fullmatch(r"HV\d+-T\d+", upper)
        ):
            return "temperature"
        return "other"

    def is_absolute_temperature(column: str) -> bool:
        upper = module.signal_name(column).upper().replace(" ", "")
        return signal_kind(column) == "temperature" and upper != "DTBR"

    module.signal_kind = signal_kind
    module.is_absolute_temperature = is_absolute_temperature


def main() -> None:
    module = load_reference_module()
    install_full88_signal_rules(module)
    module.main()


if __name__ == "__main__":
    main()
