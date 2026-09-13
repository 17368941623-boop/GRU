#!/usr/bin/env python3
"""Synthetic forward/backward test; project data are never opened."""

from __future__ import annotations

import numpy as np
import torch

from feature_protocol import (
    FEATURE_SPECS, MODEL_CONFIG, MODEL_HISTORY_COLUMNS, MODEL_NAME,
    PREDICT_STEPS, history_feature_mask,
)
from model_components import ModelConfig, build_model

CONTROLS=("CV8312","CV8311","CV8310","CV8313","CV8300","CV8351","EC-V2","COOLDOWN")


def main()->None:
    torch.manual_seed(7); counts=set()
    for spec in FEATURE_SPECS:
        model=build_model(ModelConfig(model_name=MODEL_NAME,**MODEL_CONFIG),
          MODEL_HISTORY_COLUMNS,CONTROLS,PREDICT_STEPS,
          np.zeros(len(MODEL_HISTORY_COLUMNS)),np.ones(len(MODEL_HISTORY_COLUMNS)))
        mask=torch.tensor(history_feature_mask(spec),dtype=torch.float32)
        history=torch.randn(2,60,len(MODEL_HISTORY_COLUMNS))*mask
        controls=torch.randn(2,PREDICT_STEPS,len(CONTROLS))
        output=model(history,controls)
        if output.shape!=(2,) or not torch.isfinite(output).all(): raise AssertionError(spec["feature_set"])
        output.square().mean().backward()
        if not any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()): raise AssertionError("No finite gradients")
        count=sum(p.numel() for p in model.parameters() if p.requires_grad); counts.add(count)
        print(f"SMOKE_OK={spec['feature_set']} | parameters={count}")
    if len(counts)!=1: raise AssertionError("Parameter count changed across ablations")
    print("SMOKE_TEST_COMPLETE=true")
    print("PROJECT_DATA_READ=false")


if __name__=="__main__": main()
