from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .dataset import TrainingData, build_training_data
from .model import MLPRanker, export_json_model


RL_ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the BAS learning shot ranker.")
    parser.add_argument("--config", default=str(RL_ROOT / "config.yaml"), help="Training YAML config.")
    parser.add_argument("--samples", default=None, help="Sample JSONL file or directory.")
    parser.add_argument("--out", default=None, help="Output ranker JSON path.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    samples = _resolve_path(args.samples, cfg.get("samples", "data/samples"))
    out = _resolve_path(args.out, cfg.get("output_model", "models/ranker.json"))
    epochs = int(args.epochs if args.epochs is not None else cfg.get("epochs", 50))
    batch_size = int(args.batch_size if args.batch_size is not None else cfg.get("batch_size", 128))
    lr = float(args.lr if args.lr is not None else cfg.get("learning_rate", 0.001))
    hidden_size = int(args.hidden_size if args.hidden_size is not None else cfg.get("hidden_size", 64))
    val_ratio = float(args.val_ratio if args.val_ratio is not None else cfg.get("val_ratio", 0.2))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))

    _seed_everything(seed)
    data = build_training_data(samples)
    if data.features.shape[0] == 0:
        raise SystemExit(f"No training rows found under {samples}. Enable BAS learning collection first.")

    train_idx, val_idx = _split_indices(data.features.shape[0], val_ratio, seed)
    mean = data.features[train_idx].mean(axis=0)
    std = data.features[train_idx].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x = ((data.features - mean) / std).astype(np.float32)
    y = data.target_matrix.astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPRanker(input_dim=x.shape[1], hidden_size=hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    x_tensor = torch.from_numpy(x).to(device)
    y_tensor = torch.from_numpy(y).to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_idx in _batches(train_idx, batch_size, seed + epoch):
            xb = x_tensor[batch_idx]
            yb = y_tensor[batch_idx]
            out_batch = model(xb)
            loss = _multitask_loss(out_batch, yb)
            loss = loss + 0.2 * _pairwise_rank_loss(out_batch[:, 4], yb[:, 4], [data.sample_ids[i] for i in batch_idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0:
            metrics = _evaluate(model, x_tensor, y_tensor, val_idx if len(val_idx) else train_idx)
            print(json.dumps({"epoch": epoch, "loss": float(np.mean(losses)), **metrics}, ensure_ascii=False))

    summary = {
        "rows": int(data.features.shape[0]),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "samples_path": str(samples),
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "hidden_size": hidden_size,
    }
    export_path = export_json_model(model.cpu(), out, feature_names=data.feature_names, mean=mean, std=std, training_summary=summary)
    print(json.dumps({"exported": str(export_path), **summary}, ensure_ascii=False, indent=2))
    return 0


def _load_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _resolve_path(cli_value: str | Path | None, default_value: str | Path) -> Path:
    value = cli_value if cli_value is not None else default_value
    path = Path(value)
    if path.is_absolute():
        return path
    if cli_value is not None:
        return (Path.cwd() / path).resolve()
    return (RL_ROOT / path).resolve()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_indices(n: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    val_n = int(round(n * max(0.0, min(0.8, val_ratio))))
    if n > 1:
        val_n = max(1, min(n - 1, val_n))
    else:
        val_n = 0
    return idx[val_n:], idx[:val_n]


def _batches(indices: np.ndarray, batch_size: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(indices)
    size = max(1, int(batch_size))
    return [shuffled[i : i + size] for i in range(0, len(shuffled), size)]


def _multitask_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pot = F.binary_cross_entropy_with_logits(outputs[:, 0], targets[:, 0])
    scratch = F.binary_cross_entropy_with_logits(outputs[:, 1], targets[:, 1])
    foul = F.binary_cross_entropy_with_logits(outputs[:, 2], targets[:, 2])
    leave = F.mse_loss(outputs[:, 3], targets[:, 3])
    rank = F.binary_cross_entropy_with_logits(outputs[:, 4], targets[:, 4])
    return pot + 0.6 * scratch + 0.6 * foul + 0.4 * leave + rank


def _pairwise_rank_loss(scores: torch.Tensor, relevance: torch.Tensor, group_ids: list[str]) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    groups: dict[str, list[int]] = {}
    for idx, group in enumerate(group_ids):
        groups.setdefault(group, []).append(idx)
    for indices in groups.values():
        idx = torch.as_tensor(indices, dtype=torch.long, device=scores.device)
        rel = relevance[idx]
        pos = idx[rel > 0.5]
        neg = idx[rel <= 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        diff = scores[pos].reshape(-1, 1) - scores[neg].reshape(1, -1)
        terms.append(F.softplus(-diff).mean())
    if not terms:
        return scores.sum() * 0.0
    return torch.stack(terms).mean()


@torch.no_grad()
def _evaluate(model: MLPRanker, x: torch.Tensor, y: torch.Tensor, indices: np.ndarray) -> dict[str, float]:
    model.eval()
    idx = torch.as_tensor(indices, dtype=torch.long, device=x.device)
    outputs = model(x[idx])
    target = y[idx]
    pot_prob = torch.sigmoid(outputs[:, 0])
    rank_prob = torch.sigmoid(outputs[:, 4])
    return {
        "val_pot_acc": float(((pot_prob >= 0.5) == (target[:, 0] >= 0.5)).float().mean().cpu()),
        "val_rank_acc": float(((rank_prob >= 0.5) == (target[:, 4] >= 0.5)).float().mean().cpu()),
        "val_leave_mae": float(torch.abs(outputs[:, 3] - target[:, 3]).mean().cpu()),
    }


if __name__ == "__main__":
    raise SystemExit(main())
