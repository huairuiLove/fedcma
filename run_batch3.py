"""Batch 3: retain all examples and stratum counts; tune ordering on validation."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np

from curriculum import curriculum_plan
from dataset import load_prepared, sha256
from fed import run_fed, save_json
from methods import method_components
from run_batch2 import read_json, snapshot_project, compare_states, compare_runs


def preparation(root, output, reference):
    """Never reuse an experiment directory after its protocol has been written."""
    if output.exists() and any(p.name != "preparation" for p in output.iterdir()):
        raise FileExistsError(f"Output is not a fresh preparation directory: {output}")
    prep = output / "preparation"
    prep.mkdir(parents=True, exist_ok=True)
    index = prep / "prior_batches_manifest.json"
    if not index.exists():
        batch1 = root / "experiments" / "batch_01"
        entries = [root / p for p in read_json(batch1 / "manifest.json")["entries"]]
        entries += [batch1, reference, reference.with_suffix(".console.log")]
        files = {}
        for entry in entries:
            for path in (entry.rglob("*") if entry.is_dir() else [entry]):
                if path.is_file():
                    rel = path.relative_to(root).as_posix()
                    files[rel] = {"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)}
        save_json(index, {"batches": [1, 2], "created_utc": datetime.now(timezone.utc).isoformat(),
                          "files": sorted(files.values(), key=lambda r: r["path"])})
    return index


def verify_previous(root, index):
    records = read_json(index)["files"]
    for r in records:
        path = root / r["path"]
        assert path.stat().st_size == r["bytes"] and sha256(path) == r["sha256"], str(path)
    return {"batches": [1, 2], "all_passed": True, "files_checked": len(records),
            "bytes_checked": sum(r["bytes"] for r in records), "index": str(index),
            "checked_utc": datetime.now(timezone.utc).isoformat()}


def namespace(settings):
    cfg = SimpleNamespace(**settings)
    cfg.prepared, cfg.out_dir = Path(cfg.prepared), Path(cfg.out_dir)
    return cfg


def same_training(new, old, rounds):
    for t in range(1, rounds + 1):
        compare_states(new / "checkpoints" / f"round_{t:02d}.pt", old / "checkpoints" / f"round_{t:02d}.pt")
        assert (new / f"val_round_{t:02d}.csv").read_bytes() == (old / f"val_round_{t:02d}.csv").read_bytes()
        with np.load(new / f"selection_round_{t:02d}.npz") as a, np.load(old / f"selection_round_{t:02d}.npz") as b:
            assert a.files == b.files
            for name in a.files:
                np.testing.assert_array_equal(a[name], b[name])
    compare_states(new / "best.pt", old / "best.pt")
    return {"new": str(new), "reference": str(old), "passed": True,
            "identical_all_round_weights_priors_predictions_and_training_indices": True}


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root / "configs" / "batch_03.json")
    parser.add_argument("--out_dir", type=Path)
    args = parser.parse_args()
    experiment = read_json(args.config)
    reference = (root / experiment["reference_experiment"]).resolve()
    output = (args.out_dir or root / "experiments" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("batch_03_%Y%m%d_%H%M%S")).resolve()
    index = preparation(root, output, reference)
    save_json(output / "preparation" / "prior_batches_verified_before.json", verify_previous(root, index))
    from analyze_curriculum_ap import analyze
    analyze(reference, output / "preparation" / "analysis")
    shutil.copytree(reference / "data", output / "data")
    data_hash = sha256(output / "data" / "prepared.pt")
    assert data_hash == sha256(reference / "data" / "prepared.pt")
    seeds, methods = experiment["seeds"], experiment["methods"]
    templates = {s: read_json(reference / f"fedcma_s{s}" / "config.json") for s in seeds}
    baseline_validation = {s: read_json(reference / f"fedcma_wo_cl_s{s}" / "result.json")["best_validation"]["auprc"] for s in seeds}

    def settings(seed, method, destination, overrides, validation_only):
        return {**templates[seed], **experiment["common_overrides"], **overrides,
                "method": method, "seed": seed, "prepared": str(output / "data" / "prepared.pt"),
                "out_dir": str(destination / f"{method}_s{seed}"), "validation_only": validation_only}

    study = {"batch": 3, "label": "第三批", "created_utc": datetime.now(timezone.utc).isoformat(),
             "command": [sys.executable, *sys.argv], "experiment_config": experiment,
             "reference_experiment": str(reference), "prepared_sha256": data_hash,
             "seeds": seeds, "methods": methods, "validation_control_ap": baseline_validation,
             "curriculum": "client-normalized current BCE difficulty; rank interpolation inside training label x mask strata; every sample once each epoch",
             "matching": "same random label/mask slots as the control in every mini-batch; same per-sample frequency, example count, batch sizes and optimizer steps",
             "test_policy": "all candidates validation-only; freeze selected settings before formal runs; no test-based change within batch 3",
             "development_context": "batch 1 and 2 test results have already been inspected; this is a continuous-development comparison on the same held-out patients"}
    save_json(output / "study_protocol.json", study)
    snapshot_project(root, output)
    started = time.perf_counter()
    from summarize import summarize
    from verify_results import audit
    candidates = []
    for candidate in experiment["validation_candidates"]:
        name = candidate["name"]
        overrides = {k: v for k, v in candidate.items() if k != "name"}
        destination = output / "validation_search" / name
        destination.mkdir(parents=True, exist_ok=False)
        configs = [settings(s, "fedcma", destination, overrides, True) for s in seeds]
        save_json(destination / "protocol.json", {**study, "candidate": candidate,
                  "methods": ["fedcma"], "runs": configs, "validation_only": True})
        completed = []
        for config in configs:
            completed.append(run_fed(namespace(config)))
            save_json(destination / "completed_runs.json", completed)
        summarize(destination)
        audit(destination)
        assert not list(destination.glob("*/test_predictions.csv"))
        aps = [r["best_validation"]["auprc"] for r in completed]
        gains = [a - baseline_validation[s] for a, s in zip(aps, seeds)]
        candidates.append({"name": name, "overrides": overrides, "validation_ap": aps,
                           "paired_gains": gains, "minimum_gain": min(gains),
                           "mean_gain": float(np.mean(gains)), "gain_std": float(np.std(gains, ddof=1)),
                           "positive_gain_seeds": sum(g > 0 for g in gains)})
        save_json(output / "validation_candidates_completed.json", candidates)
    winner = max(candidates, key=lambda r: (r["minimum_gain"], r["mean_gain"]))
    save_json(output / "candidate_selection.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
              "rule": experiment["candidate_selection"], "seeds": seeds, "validation_control_ap": baseline_validation,
              "candidates": candidates, "selected": winner, "candidate_test_evaluations": 0})
    configs = [settings(s, m, output, winner["overrides"], False) for s in seeds for m in methods]
    train = load_prepared(output / "data" / "prepared.pt", "cpu")[0]["train"]
    max_missing = float((1 - train["masks"].mean(1)).max())
    cfg = namespace(configs[0])
    schedule = [curriculum_plan(cfg, t, True, max_missing) for t in range(cfg.rounds)]
    del train
    save_json(output / "protocol.json", {**study, "formal_created_utc": datetime.now(timezone.utc).isoformat(),
              "selected_candidate": winner, "curriculum_schedule": schedule,
              "components": {m: method_components(m) for m in methods}, "ablations": True, "runs": configs})
    completed = []
    for config in configs:
        completed.append(run_fed(namespace(config)))
        save_json(output / "completed_runs.json", completed)
    summarize(output)
    verification = audit(output)
    assert len({(r["training_examples"], r["step_records"]) for r in verification["runs"]}) == 1
    controls, warmups, pilot_replays, slots = [], [], [], []
    for seed in seeds:
        for method in ("fedavg", "fedprox", "fedcma_wo_cl", "fedcma_wo_both"):
            controls.append(compare_runs(output / f"{method}_s{seed}", reference / f"{method}_s{seed}", cfg.rounds))
        controls.append(compare_runs(output / f"fedcma_wo_both_s{seed}", output / f"fedavg_s{seed}", cfg.rounds))
        pilot_replays.append(same_training(output / f"fedcma_s{seed}",
                             output / "validation_search" / winner["name"] / f"fedcma_s{seed}", cfg.rounds))
        for a, b in (("fedcma", "fedcma_wo_cl"), ("fedcma_wo_prior", "fedavg")):
            for t in range(1, cfg.warmup_rounds + 1):
                compare_states(output / f"{a}_s{seed}" / "checkpoints" / f"round_{t:02d}.pt",
                               output / f"{b}_s{seed}" / "checkpoints" / f"round_{t:02d}.pt")
            warmups.append({"method": a, "control": b, "seed": seed, "passed": True})
            for t in range(1, cfg.rounds + 1):
                with np.load(output / f"{a}_s{seed}" / f"selection_round_{t:02d}.npz") as x, \
                     np.load(output / f"{b}_s{seed}" / f"selection_round_{t:02d}.npz") as y:
                    for k in range(cfg.num_clients):
                        for e in range(1, cfg.local_epochs + 1):
                            np.testing.assert_array_equal(x[f"client_{k}_epoch_{e}_random"], y[f"client_{k}_epoch_{e}_train"])
            slots.append({"method": a, "control": b, "seed": seed, "all_random_slots_match_control": True})
    save_json(output / "control_verification.json", {"all_passed": True, "comparisons": controls,
              "warmup_comparisons": warmups, "selected_pilot_replays": pilot_replays, "random_slot_comparisons": slots})
    replay_settings = next(c for c in configs if c["method"] == "fedcma" and c["seed"] == seeds[0])
    replay = namespace(replay_settings)
    replay.out_dir = output / "retraining_check" / f"fedcma_s{seeds[0]}"
    run_fed(replay)
    replay_checks = same_training(replay.out_dir, output / f"fedcma_s{seeds[0]}", cfg.rounds)
    compare_runs(replay.out_dir, output / f"fedcma_s{seeds[0]}", cfg.rounds)
    save_json(output / "retraining_verification.json", {"all_passed": True, **replay_checks})
    save_json(output / "prior_batches_preservation_after.json", verify_previous(root, index))
    save_json(output / "execution_summary.json", {"n_experiment_runs": len(completed),
              "n_validation_runs": len(candidates) * len(seeds), "n_extra_retraining_checks": 1,
              "end_utc": datetime.now(timezone.utc).isoformat(), "total_seconds_including_audits": time.perf_counter() - started})
    from report_batch3 import report
    report(output)
    print(f"BATCH 3 COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
