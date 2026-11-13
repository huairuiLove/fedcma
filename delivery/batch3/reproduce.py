"""Portable batch-3 reproduction using the frozen source and recorded protocols."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace


BATCH = Path("records/experiments/batch_03_ihm_20261113")
MAIN = {"fedavg", "fedprox", "fedcma"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def selected_runs(archive, suite, seeds, methods):
    entries = []
    if suite in ("main", "formal", "all"):
        for cfg in read(archive / "protocol.json")["runs"]:
            if suite != "main" or cfg["method"] in MAIN:
                entries.append((Path(f"{cfg['method']}_s{cfg['seed']}"), cfg))
    if suite in ("search", "all"):
        for path in sorted((archive / "validation_search").glob("*/protocol.json")):
            for cfg in read(path)["runs"]:
                entries.append((path.parent.relative_to(archive) / f"{cfg['method']}_s{cfg['seed']}", cfg))
    if suite == "all":
        rel = Path("retraining_check/fedcma_s42")
        entries.append((rel, read(archive / rel / "config.json")))
    return [(rel, cfg) for rel, cfg in entries
            if cfg["seed"] in seeds and (not methods or cfg["method"] in methods)]


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.stream.write(value)
        self.log.write(value)

    def flush(self):
        self.stream.flush()
        self.log.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--suite", choices=("main", "formal", "search", "all"), default="main")
    parser.add_argument("--seeds", type=int, nargs="+", choices=(42, 43, 44), default=[42, 43, 44])
    parser.add_argument("--methods", nargs="+", choices=("fedavg", "fedprox", "fedcma", "fedcma_wo_prior", "fedcma_wo_cl", "fedcma_wo_both"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", help="Omit to retain the recorded CUDA setting; cpu is also supported.")
    parser.add_argument("--compare", action="store_true", help="Require exactly matching weights, predictions and sample order.")
    args = parser.parse_args()
    root = args.package.resolve()
    archive = root / BATCH
    entries = selected_runs(archive, args.suite, args.seeds, args.methods)
    if not entries:
        parser.error("No runs match these filters")
    output = (args.output or root / "reproduced" / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    if output == root or any(output.is_relative_to(root / p) for p in ("records", "code", "input_data", "patent")):
        parser.error("Output must be separate from the archived inputs and records")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    output.mkdir(parents=True)
    shutil.copytree(archive / "data", output / "data")
    sys.path.insert(0, str(root / "code"))
    from fed import run_fed, save_json
    from evaluate_checkpoint import verify
    from run_batch3 import same_training
    from dataset import sha256
    for name, value in read(archive / "project_sha256.json").items():
        if sha256(root / "code" / name) != value:
            raise ValueError(f"Frozen source hash mismatch: {name}")
    groups, planned = {}, []
    for rel, original in entries:
        cfg = dict(original, prepared=str(output / "data/prepared.pt"), out_dir=str(output / rel))
        if args.device:
            cfg["device"] = args.device
        groups.setdefault(rel.parent, []).append(cfg)
        planned.append((rel, cfg))
    for parent, configs in groups.items():
        (output / parent).mkdir(parents=True, exist_ok=True)
        save_json(output / parent / "protocol.json", {"batch": 3, "runs": configs})
    save_json(output / "reproduction_plan.json", {"batch": 3, "suite": args.suite,
              "archive": str(archive), "runs": [str(rel) for rel, _ in planned],
              "device_override": args.device, "source": "frozen batch-3 project_snapshot"})
    reports = []
    with (output / "console.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            for rel, cfg in planned:
                print(f"Reproducing {rel}", flush=True)
                settings = SimpleNamespace(**cfg)
                settings.prepared, settings.out_dir = Path(cfg["prepared"]), Path(cfg["out_dir"])
                run_fed(settings)
                replay = verify(output / rel, settings.prepared, cfg["device"])
                record = {"run": rel.as_posix(), "checkpoint_replay": replay}
                if args.compare:
                    record["original_comparison"] = same_training(output / rel, archive / rel, cfg["rounds"])
                    if not cfg["validation_only"]:
                        assert (output / rel / "test_predictions.csv").read_bytes() == (archive / rel / "test_predictions.csv").read_bytes()
                    record["original_comparison"]["test_predictions_identical"] = not cfg["validation_only"]
                reports.append(record)
                save_json(output / "reproduction_verification.json", {"all_passed": True, "completed_runs": len(reports), "runs": reports})
    print(f"Completed {len(reports)} runs. Records: {output}")


if __name__ == "__main__":
    main()
