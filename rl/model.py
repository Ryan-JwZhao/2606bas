from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn


OUTPUT_NAMES = ["pot_logit", "scratch_logit", "foul_logit", "leave_score", "rank_score"]


class MLPRanker(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 64, output_dim: int = len(OUTPUT_NAMES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def export_json_model(
    model: MLPRanker,
    out_path: str | Path,
    *,
    feature_names: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    training_summary: dict[str, Any],
) -> Path:
    layers = []
    for module in model.net:
        if isinstance(module, nn.Linear):
            layers.append(
                {
                    "weight": module.weight.detach().cpu().numpy().astype(float).tolist(),
                    "bias": module.bias.detach().cpu().numpy().astype(float).tolist(),
                }
            )
    payload = {
        "format": "bas_mlp_ranker_v1",
        "model_version": f"ranker_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "feature_names": list(feature_names),
        "normalization": {
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
        },
        "output_names": OUTPUT_NAMES,
        "score_weights": {
            "pot": 1.2,
            "scratch": 1.1,
            "foul": 0.9,
            "leave": 0.4,
            "rank": 0.8,
            "risk": 0.5,
            "residual": 0.0,
        },
        "layers": layers,
        "training": training_summary,
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
