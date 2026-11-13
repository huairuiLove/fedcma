"""Verify archive hashes and optionally replay all 31 original best checkpoints."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, help="Write a new verification report outside the frozen archive.")
    args = parser.parse_args()
    root = args.package.resolve()
    archive = root / "records/experiments/batch_03_ihm_20261113"
    original = read(archive / "artifacts_manifest.json")
    for item in original["files"]:
        path = root / "records" / item["path"]
        assert path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"], str(path)
    for name, digest in read(archive / "project_sha256.json").items():
        assert sha256(root / "code" / name) == digest, name
    data = read(archive / "data/manifest.json")
    for name, digest in data["input_sha256"].items():
        assert sha256(root / "input_data/ihm" / name) == digest, name
    assert sha256(archive / "data/prepared.pt") == data["prepared_sha256"]
    all_files = root / "package_manifest.json"
    package_count = 0
    if all_files.exists():
        for item in read(all_files)["files"]:
            path = root / item["path"]
            assert path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"], str(path)
            package_count += 1
    runs = sorted(archive.rglob("result.json"))
    assert len(runs) == 31
    step_count = sum(sum(1 for _ in (p.parent / "steps.jsonl").open(encoding="utf-8")) for p in runs)
    checkpoint_count = sum(len(list((p.parent / "checkpoints").glob("round_*.pt"))) for p in runs)
    assert (step_count, checkpoint_count) == (23250, 155)
    reports = []
    if args.replay:
        sys.path.insert(0, str(root / "code"))
        from evaluate_checkpoint import verify
        for path in runs:
            reports.append(verify(path.parent, archive / "data/prepared.pt", args.device))
    report = {"all_passed": True, "checked_utc": datetime.now(timezone.utc).isoformat(),
              "archive_files_checked": len(original["files"]), "package_files_checked": package_count,
              "original_runs": len(runs), "step_records": step_count, "round_checkpoints": checkpoint_count,
              "best_models": len(runs), "input_and_source_hashes_verified": True,
              "checkpoint_replays": reports}
    if args.output:
        output = args.output.resolve()
        if output.is_relative_to(root / "records") or output.exists():
            raise ValueError("Verification output must be a new file outside the frozen records")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "checkpoint_replays"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
