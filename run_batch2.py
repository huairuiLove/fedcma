"""Batch 2: fixed warmup curriculum, matched training budgets, all six controls."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import torch

from curriculum import curriculum_plan
from dataset import load_prepared, sha256
from fed import run_fed, save_json
from methods import method_components


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_archive(root, index):
    manifest = read_json(index)
    for record in manifest["files"]:
        path = root / record["path"]
        assert path.stat().st_size == record["bytes"], str(path)
        assert sha256(path) == record["sha256"], str(path)
    return {"batch": 1, "all_passed": True, "files_checked": len(manifest["files"]),
            "bytes_checked": sum(r["bytes"] for r in manifest["files"]),
            "index": str(index), "checked_utc": datetime.now(timezone.utc).isoformat()}


def compare_states(new_path, old_path):
    new = torch.load(new_path, map_location="cpu", weights_only=True)
    old = torch.load(old_path, map_location="cpu", weights_only=True)
    assert new["round"] == old["round"]
    assert new["model"].keys() == old["model"].keys()
    assert all(torch.equal(v, old["model"][k]) for k, v in new["model"].items()), str(new_path)
    for key in ("priors", "reference_prior"):
        assert torch.equal(new[key], old[key]), f"{new_path}: {key}"
    assert new["validation"] == old["validation"]


def compare_runs(new, old, rounds):
    for t in range(1, rounds + 1):
        compare_states(new / "checkpoints" / f"round_{t:02d}.pt",
                       old / "checkpoints" / f"round_{t:02d}.pt")
        assert (new / f"val_round_{t:02d}.csv").read_bytes() == (old / f"val_round_{t:02d}.csv").read_bytes()
    compare_states(new / "best.pt", old / "best.pt")
    assert (new / "test_predictions.csv").read_bytes() == (old / "test_predictions.csv").read_bytes()
    return {"new_run": str(new), "reference_run": str(old), "passed": True,
            "rounds_identical": rounds, "identical_model_and_priors": True,
            "identical_validation_and_test_predictions": True}


def snapshot_project(root, output):
    snapshot = output / "project_snapshot"
    snapshot.mkdir()
    for path in root.iterdir():
        if path.is_file() and path.suffix in (".py", ".ps1", ".sh", ".md", ".txt"):
            shutil.copy2(path, snapshot / path.name)
    for name in ("configs", "tests"):
        (snapshot / name).mkdir()
        for path in (root / name).iterdir():
            if path.is_file():
                shutil.copy2(path, snapshot / name / path.name)
    save_json(output / "project_sha256.json", {
        p.relative_to(snapshot).as_posix(): sha256(p) for p in sorted(snapshot.rglob("*")) if p.is_file()})


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root / "configs" / "batch_02.json")
    parser.add_argument("--out_dir", type=Path)
    args = parser.parse_args()
    experiment = read_json(args.config)
    reference = (root / experiment["reference_main"]).resolve()
    ablation = (root / experiment["reference_ablation"]).resolve()
    index = (root / experiment["batch1_index"]).resolve()
    output = (args.out_dir or root / "experiments" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("batch_02_%Y%m%d_%H%M%S")).resolve()
    seeds, methods = experiment["seeds"], experiment["methods"]
    preservation_before = verify_archive(root, index)
    configs = []
    for seed in seeds:
        old = read_json(reference / f"fedcma_s{seed}" / "config.json")
        assert not old["validation_only"] and old["seed"] == seed
        for method in methods:
            method_components(method)
            configs.append({**old, **experiment["overrides"], "method": method,
                            "prepared": str(output / "data" / "prepared.pt"),
                            "out_dir": str(output / f"{method}_s{seed}")})
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(reference / "data", output / "data")
    prepared_hash = sha256(output / "data" / "prepared.pt")
    assert prepared_hash == sha256(reference / "data" / "prepared.pt")
    train = load_prepared(output / "data" / "prepared.pt", "cpu")[0]["train"]
    max_missing = float((1 - train["masks"].mean(1)).max())
    cfg = SimpleNamespace(**configs[0])
    schedule = [curriculum_plan(cfg, t, True, max_missing) for t in range(cfg.rounds)]
    del train
    save_json(output / "protocol.json", {
        "batch": 2, "label": "第二批", "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "experiment_config": experiment,
        "reference_experiment": str(reference), "reference_ablation": str(ablation),
        "batch1_index": str(index), "prepared_sha256": prepared_hash,
        "methods": methods, "seeds": seeds, "ablations": True,
        "new_hyperparameter_search": False, "components": {m: method_components(m) for m in methods},
        "curriculum_schedule": schedule,
        "difficulty": "xi * missing modality fraction + (1 - xi) * current BCE / scoring-batch max BCE",
        "warmup": "all samples; no difficulty scoring; counts inside the same five total rounds",
        "threshold_bound": "xi * max training-mask missing fraction + (1 - xi); no validation/test calibration",
        "budget": "each client draws its original n_k examples per epoch, shuffled without replacement per cycle; repeat eligible pool as needed",
        "budget_scope": "same training examples, batch sizes and optimizer steps; curriculum has additional scoring forwards",
        "aggregation": "original n_k weights, unchanged from batch 1",
        "selection_metric": "global_validation_average_precision",
        "test_policy": "fixed single configuration before runs; validation chooses checkpoint; retain all seeds and methods",
        "runs": configs})
    save_json(output / "batch1_preservation_before.json", preservation_before)
    snapshot_project(root, output)
    completed, start = [], time.perf_counter()
    for settings in configs:
        cfg = SimpleNamespace(**settings)
        cfg.prepared, cfg.out_dir = Path(cfg.prepared), Path(cfg.out_dir)
        completed.append(run_fed(cfg))
        save_json(output / "completed_runs.json", completed)
    from summarize import summarize
    summarize(output)
    from verify_results import audit
    verification = audit(output)
    assert len({(r["training_examples"], r["step_records"]) for r in verification["runs"]}) == 1
    comparisons, warmups = [], []
    for seed in seeds:
        for method, old_root in (("fedavg", reference), ("fedprox", reference),
                                 ("fedcma_wo_cl", ablation), ("fedcma_wo_both", ablation)):
            comparisons.append(compare_runs(output / f"{method}_s{seed}", old_root / f"{method}_s{seed}", cfg.rounds))
        comparisons.append(compare_runs(output / f"fedcma_wo_both_s{seed}", output / f"fedavg_s{seed}", cfg.rounds))
        for a, b in (("fedcma", "fedcma_wo_cl"), ("fedcma_wo_prior", "fedavg")):
            for t in range(1, cfg.warmup_rounds + 1):
                compare_states(output / f"{a}_s{seed}" / "checkpoints" / f"round_{t:02d}.pt",
                               output / f"{b}_s{seed}" / "checkpoints" / f"round_{t:02d}.pt")
            warmups.append({"method": a, "control": b, "seed": seed,
                            "identical_warmup_rounds": cfg.warmup_rounds, "passed": True})
    save_json(output / "control_verification.json", {"all_passed": True, "comparisons": comparisons, "warmup_comparisons": warmups})
    # One full rerun verifies deterministic selection/repetition and training.
    settings = next(r for r in configs if r["method"] == "fedcma" and r["seed"] == seeds[0])
    replay = SimpleNamespace(**settings)
    replay.prepared = Path(replay.prepared)
    replay.out_dir = output / "retraining_check" / f"fedcma_s{seeds[0]}"
    run_fed(replay)
    source_run = output / f"fedcma_s{seeds[0]}"
    replay_check = compare_runs(replay.out_dir, source_run, replay.rounds)
    for t in range(1, replay.rounds + 1):
        assert sha256(replay.out_dir / f"selection_round_{t:02d}.npz") == sha256(source_run / f"selection_round_{t:02d}.npz")
    save_json(output / "retraining_verification.json", {"all_passed": True, "identical_selections_and_training_indices": True, **replay_check})
    save_json(output / "batch1_preservation_after.json", verify_archive(root, index))
    save_json(output / "execution_summary.json", {"n_experiment_runs": len(completed), "n_extra_retraining_checks": 1,
              "end_utc": datetime.now(timezone.utc).isoformat(), "total_seconds_including_audits": time.perf_counter() - start})
    from report_batch2 import report
    report(output)
    print(f"BATCH 2 COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
