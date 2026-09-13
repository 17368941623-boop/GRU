"""Frozen protocol for the 0912 Raw54-GRU baseline."""

from __future__ import annotations

from pathlib import Path


SAMPLE_PERIOD_SECONDS = 10
LOOKBACK = 60
COMMON_ORIGIN_LOOKBACK = 120
PREDICT_STEPS = 15
TARGET = "Thv"
SEEDS = (42, 52, 62, 72, 82, 92, 102, 112, 122, 132)

TRAIN_PARENTS = (
    "251130",
    "251226",
    "260118",
    "260403",
    "260617",
    "260623-BACK",
    "260715",
)
VALIDATION_PARENT = "260428"
FROZEN_TEST_PARENT = "260721-BACK"

ALIASES = {
    "THV": "Thv",
    "TEF": "Tef",
    "TCD": "Tcd",
    "DTBR": "DTbr",
}

FUTURE_CONTROL_COLUMNS = (
    "CV8312",
    "CV8311",
    "CV8310",
    "CV8313",
    "CV8300",
    "CV8351",
    "EC-V2",
    "COOLDOWN",
)


def project_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def default_source_dir() -> Path:
    return Path(r"C:\Users\Administrator\Desktop\新建文件夹\filtered_data")


def cache_dir() -> Path:
    return project_dir() / "data_cache"


def output_dir() -> Path:
    return project_dir() / "outputs"


def run_dir(seed: int) -> Path:
    return output_dir() / "development" / "raw54_gru" / f"seed_{seed}"
