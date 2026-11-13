"""Fixed-setting 2x2 ablation using the completed main experiment as reference."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import torch

from dataset import sha256
from fed import run_fed, save_json
from methods import method_components


ABLATION_METHODS = ["fedcma", "fedcma_wo_prior", "fedcma_wo_cl", "fedcma_wo_both"]


def compare_reference(new_run, old_run):
    """Ensure controls reproduce the earlier full model and FedAvg exactly."""
    new = torch.load(new_run / "best.pt", map_location="cpu", weights_only=True)
    old = torch.load(old_run / "best.pt", map_location="cpu", weights_only=True)
    assert new["round"] == old["round"]
    assert all(torch.equal(value, old["model"][key]) for key, value in new["model"].items())
    assert torch.equal(new["priors"], old["priors"])
    assert torch.equal(new["reference_prior"], old["reference_prior"])
    assert (new_run / "test_predictions.csv").read_bytes() == (old_run / "test_predictions.csv").read_bytes()
    return {"new_run": str(new_run), "reference_run": str(old_run), "passed": True,
            "identical_model_tensors": len(new["model"]), "identical_predictions": True}


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_dir", type=Path, default=root / "experiments" / "final_ihm_20261113")
    parser.add_argument("--out_dir", type=Path)
    args = parser.parse_args()
    reference = args.reference_dir.resolve()
    output = (args.out_dir or root / "experiments" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("ablation_%Y%m%d_%H%M%S")).resolve()
    reference_protocol = json.loads((reference / "protocol.json").read_text(encoding="utf-8"))
    seeds = reference_protocol["seeds"]
    configs = []
    # Copy exact per-seed settings, with only method and output/data location changed.
    for seed in seeds:
        settings = json.loads((reference / f"fedcma_s{seed}" / "config.json").read_text(encoding="utf-8"))
        assert settings["seed"] == seed and not settings["validation_only"]
        for method in ABLATION_METHODS:
            configs.append({**settings, "method": method,
                            "prepared": str(output / "data" / "prepared.pt"),
                            "out_dir": str(output / f"{method}_s{seed}")})
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(reference / "data", output / "data")
    assert sha256(output / "data" / "prepared.pt") == sha256(reference / "data" / "prepared.pt")
    save_json(output / "protocol.json", {
        "command": [sys.executable, *sys.argv], "reference_experiment": str(reference),
        "followup_ablation": True, "ablations": True, "new_hyperparameter_search": False,
        "methods": ABLATION_METHODS, "seeds": seeds,
        "components": {m: method_components(m) for m in ABLATION_METHODS},
        "prior_removal": "disable learned priors, generators, alignment, prior regularization and sharing; use zero missing features",
        "curriculum_removal": "use every local training sample every round; retain the same epochs and LR",
        "selection_metric": "global_validation_average_precision",
        "test_policy": "only validation selects checkpoints; no tuning after ablation test results",
        "runs": configs})
    completed = []
    for settings in configs:
        cfg = SimpleNamespace(**settings)
        cfg.prepared, cfg.out_dir = Path(cfg.prepared), Path(cfg.out_dir)
        completed.append(run_fed(cfg))
        save_json(output / "completed_runs.json", completed)
    from summarize import summarize
    summarize(output)
    from verify_results import audit
    audit(output)
    controls = []
    for seed in seeds:
        controls.append(compare_reference(output / f"fedcma_s{seed}", reference / f"fedcma_s{seed}"))
        controls.append(compare_reference(output / f"fedcma_wo_both_s{seed}", reference / f"fedavg_s{seed}"))
    save_json(output / "control_verification.json", {"all_passed": True, "comparisons": controls})
    from report_ablation import report
    report(output, output / "ablation_report.md")
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
    print(f"ABLATION COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
