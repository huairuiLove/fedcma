"""Recompute saved checkpoint predictions and check against the recorded results."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from dataset import load_prepared
from fed import evaluate, seed_everything
from model import FedCMAModel
from methods import method_components


def verify(run_dir, prepared=None, device=None):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    cfg = SimpleNamespace(**config)
    device = device or cfg.device
    seed_everything(cfg.seed, cfg.threads)
    adjacent_data = run_dir.parent / "data" / "prepared.pt"
    data_path = prepared or (adjacent_data if adjacent_data.exists() else cfg.prepared)
    data, manifest = load_prepared(data_path, device)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    if checkpoint["prepared_sha256"] != manifest["prepared_sha256"]:
        raise ValueError("Checkpoint and evaluation data differ")
    model = FedCMAModel(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    prior = checkpoint["reference_prior"] if method_components(cfg.method)["prior"] else None
    split = "val" if cfg.validation_only else "test"
    metrics, scores = evaluate(model, data[split], prior)
    saved_file = run_dir / (f"val_round_{checkpoint['round']:02d}.csv" if cfg.validation_only else "test_predictions.csv")
    with saved_file.open(encoding="utf-8") as stream:
        saved = np.array([float(row["probability"]) for row in csv.DictReader(stream)])
    difference = float(np.max(np.abs(saved - scores)))
    if not np.allclose(saved, scores, rtol=0, atol=1e-6):
        raise AssertionError(f"Predictions not reproduced: max difference {difference}")
    result = {"run": str(run_dir), "split": split, "metrics": metrics,
              "maximum_prediction_difference": difference, "passed": True}
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()
    verify(args.run_dir, args.prepared, args.device)
