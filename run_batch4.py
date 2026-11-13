"""Batch 4: bounded online weighting with unchanged random training batches."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np

from curriculum import curriculum_plan
from dataset import sha256
from fed import run_fed, save_json
from methods import method_components
from run_batch2 import read_json, snapshot_project, compare_states, compare_runs
from run_batch3 import namespace, same_training


def preparation(root, output, reference):
    if output.exists() and any(p.name != "preparation" for p in output.iterdir()):
        raise FileExistsError(f"Output already contains experiment artifacts: {output}")
    prep = output / "preparation"
    prep.mkdir(parents=True, exist_ok=True)
    index = prep / "prior_batches_manifest.json"
    if not index.exists():
        prior = read_json(reference / "preparation" / "prior_batches_manifest.json")
        files = {r["path"]: r for r in prior["files"]}
        for path in [*reference.rglob("*"), reference.with_suffix(".console.log")]:
            if path.is_file():
                rel = path.relative_to(root).as_posix()
                files[rel] = {"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)}
        save_json(index, {"batches": [1, 2, 3], "created_utc": datetime.now(timezone.utc).isoformat(),
                          "files": sorted(files.values(), key=lambda r: r["path"])})
    return index


def verify_previous(root, index):
    manifest = read_json(index)
    for r in manifest["files"]:
        path = root / r["path"]
        assert path.stat().st_size == r["bytes"] and sha256(path) == r["sha256"], str(path)
    return {"batches": manifest["batches"], "all_passed": True, "files_checked": len(manifest["files"]),
            "bytes_checked": sum(r["bytes"] for r in manifest["files"]), "index": str(index),
            "checked_utc": datetime.now(timezone.utc).isoformat()}


def validation_measures(run):
    result, history = read_json(run / "result.json"), read_json(run / "history.json")
    return {"best": result["best_validation"]["auprc"],
            "last_two_mean": float(np.mean([r["val"]["auprc"] for r in history[-2:]]))}


def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=root / "configs" / "batch_04.json")
    p.add_argument("--out_dir", type=Path)
    args = p.parse_args()
    experiment = read_json(args.config)
    reference = (root / experiment["reference_experiment"]).resolve()
    output = (args.out_dir or root / "experiments" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("batch_04_%Y%m%d_%H%M%S")).resolve()
    index = preparation(root, output, reference)
    save_json(output / "preparation" / "prior_batches_verified_before.json", verify_previous(root, index))
    from analyze_batch3_curriculum import analyze
    analyze(reference, output / "preparation" / "analysis")
    shutil.copytree(reference / "data", output / "data")
    data_hash = sha256(output / "data" / "prepared.pt")
    assert data_hash == sha256(reference / "data" / "prepared.pt")
    seeds, methods = experiment["seeds"], experiment["methods"]
    templates = {s: read_json(reference / f"fedcma_s{s}" / "config.json") for s in seeds}
    baseline = {s: validation_measures(reference / f"fedcma_wo_cl_s{s}") for s in seeds}

    def settings(seed, method, destination, overrides, validation_only):
        return {**templates[seed], **experiment["common_overrides"], **overrides,
                "method": method, "seed": seed, "prepared": str(output / "data" / "prepared.pt"),
                "out_dir": str(destination / f"{method}_s{seed}"), "validation_only": validation_only}

    study = {"batch": 4, "label": "第四批", "created_utc": datetime.now(timezone.utc).isoformat(),
             "command": [sys.executable, *sys.argv], "experiment_config": experiment,
             "reference_experiment": str(reference), "prepared_sha256": data_hash,
             "seeds": seeds, "methods": methods, "validation_control": baseline,
             "curriculum": "w = 1 + alpha_t * centered average BCE rank inside each training-batch label x mask group; weights detached; alpha ramps linearly after warmup",
             "matching": "same ordered training indices and batches as no-curriculum; each sample once per epoch; each label x mask group has mean weight one; same update and example budgets",
             "loss": "weighted task BCE plus unchanged alignment/prior regularizers; no extra ranking loss; baselines use original loss exactly",
             "logging": "all step losses, unweighted BCE, sample weights and their current per-example BCE, actual training indices and every round checkpoint",
             "test_policy": "six predefined candidates validation-only; selected settings frozen before formal runs; no candidate test evaluation or test-based change within batch 4",
             "development_context": "previous three batches used this same held-out patient split; current comparison is continued development, not a new independent test dataset"}
    save_json(output / "study_protocol.json", study)
    snapshot_project(root, output)
    started = time.perf_counter()
    from summarize import summarize
    from verify_results import audit
    candidates = []
    for candidate in experiment["validation_candidates"]:
        overrides = {k: v for k, v in candidate.items() if k != "name"}
        destination = output / "validation_search" / candidate["name"]
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
        measures = [validation_measures(destination / f"fedcma_s{s}") for s in seeds]
        best_gains = [v["best"] - baseline[s]["best"] for s, v in zip(seeds, measures)]
        late_gains = [v["last_two_mean"] - baseline[s]["last_two_mean"] for s, v in zip(seeds, measures)]
        candidates.append({"name": candidate["name"], "overrides": overrides, "validation_measures": measures,
                           "best_ap_gains": best_gains, "late_ap_gains": late_gains,
                           "stability_floor": min(best_gains + late_gains),
                           "combined_mean_gain": float(np.mean(best_gains + late_gains)),
                           "best_ap_positive_seeds": sum(g > 0 for g in best_gains),
                           "late_ap_positive_seeds": sum(g > 0 for g in late_gains)})
        save_json(output / "validation_candidates_completed.json", candidates)
    winner = max(candidates, key=lambda r: (r["stability_floor"], r["combined_mean_gain"]))
    save_json(output / "candidate_selection.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
              "rule": experiment["candidate_selection"], "seeds": seeds, "validation_control": baseline,
              "candidates": candidates, "selected": winner, "candidate_test_evaluations": 0})
    configs = [settings(s, m, output, winner["overrides"], False) for s in seeds for m in methods]
    cfg = namespace(configs[0])
    schedule = [curriculum_plan(cfg, t, True, .5) for t in range(cfg.rounds)]
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
    controls, warmups, replays, orders = [], [], [], []
    for seed in seeds:
        for method in ("fedavg", "fedprox", "fedcma_wo_cl", "fedcma_wo_both"):
            controls.append(compare_runs(output / f"{method}_s{seed}", reference / f"{method}_s{seed}", cfg.rounds))
        controls.append(compare_runs(output / f"fedcma_wo_both_s{seed}", output / f"fedavg_s{seed}", cfg.rounds))
        replays.append(same_training(output / f"fedcma_s{seed}", output / "validation_search" / winner["name"] / f"fedcma_s{seed}", cfg.rounds))
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
                            np.testing.assert_array_equal(x[f"client_{k}_epoch_{e}_train"], y[f"client_{k}_epoch_{e}_train"])
            orders.append({"method": a, "control": b, "seed": seed, "identical_training_order_every_batch": True})
    save_json(output / "control_verification.json", {"all_passed": True, "comparisons": controls,
              "warmup_comparisons": warmups, "selected_pilot_replays": replays, "training_order_comparisons": orders})
    replay = namespace(next(c for c in configs if c["method"] == "fedcma" and c["seed"] == seeds[0]))
    replay.out_dir = output / "retraining_check" / f"fedcma_s{seeds[0]}"
    run_fed(replay)
    replay_checks = same_training(replay.out_dir, output / f"fedcma_s{seeds[0]}", cfg.rounds)
    compare_runs(replay.out_dir, output / f"fedcma_s{seeds[0]}", cfg.rounds)
    save_json(output / "retraining_verification.json", {"all_passed": True, **replay_checks})
    save_json(output / "prior_batches_preservation_after.json", verify_previous(root, index))
    save_json(output / "execution_summary.json", {"n_experiment_runs": len(completed),
              "n_validation_runs": len(candidates) * len(seeds), "n_extra_retraining_checks": 1,
              "end_utc": datetime.now(timezone.utc).isoformat(), "total_seconds_including_audits": time.perf_counter() - started})
    from report_batch4 import report
    report(output)
    print(f"BATCH 4 COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
