"""Assemble only batch 3, preserving every indexed artifact without modification."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def copy(source, target, expected=None):
    if expected and sha256(source) != expected:
        raise ValueError(f"Source hash mismatch: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if expected and sha256(target) != expected:
        raise ValueError(f"Copied file hash mismatch: {target}")


def assemble(repo, output):
    if output.exists():
        raise FileExistsError(f"Package directory already exists: {output}")
    output.mkdir(parents=True)
    archive = repo / "experiments/batch_03_ihm_20261113"
    manifest = read(archive / "artifacts_manifest.json")
    for item in manifest["files"]:
        copy(repo / item["path"], output / "records" / item["path"], item["sha256"])
    copy(archive / "artifacts_manifest.json", output / "records/experiments/batch_03_ihm_20261113/artifacts_manifest.json")
    for name, digest in read(archive / "project_sha256.json").items():
        copy(archive / "project_snapshot" / name, output / "code" / name, digest)
    for path in sorted((repo / "data/ihm").iterdir()):
        if path.is_file():
            copy(path, output / "input_data/ihm" / path.name)
    for name in ("reproduce.py", "verify_package.py", "rebuild_data.py", "README.md"):
        copy(Path(__file__).parent / name, output / name)
    summary = [r for r in read(archive / "summary.json") if r["method"] in ("fedavg", "fedprox", "fedcma")]
    save(output / "main_results/summary.json", summary)
    with (output / "main_results/summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (archive / "per_seed.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = [r for r in reader if r["method"] in ("fedavg", "fedprox", "fedcma")]
        fields = reader.fieldnames
    with (output / "main_results/per_seed.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    revision = subprocess.check_output(["git", "-c", f"safe.directory={repo.as_posix()}", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    save(output / "package_info.json", {"batch": 3, "label": "第三批交付", "source_revision_at_assembly": revision,
         "branch": "delivery/batch3-patent-20261113", "repository": "https://github.com/huairuiLove/fedcma",
         "training_code": "code/ is an exact copy of the original batch-3 project_snapshot",
         "experiment_records": "records/ contains all indexed batch-3 artifacts, unchanged",
         "main_methods": ["fedavg", "fedprox", "fedcma"], "seeds": [42, 43, 44],
         "original_training_runs": 31, "original_step_records": 23250, "original_round_checkpoints": 155,
         "original_best_models": 31, "included_batches": [3],
         "created_utc": datetime.now(timezone.utc).isoformat()})
    print(f"Assembled {len(manifest['files'])} original indexed artifacts at {output}")


def finalize(root, zip_path):
    excluded = {"package_manifest.json"}
    files = [p for p in sorted(root.rglob("*")) if p.is_file()
             and p.relative_to(root).as_posix() not in excluded
             and not any(part in ("__pycache__", ".pytest_cache") for part in p.parts)]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "files": [
        {"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]}
    save(root / "package_manifest.json", manifest)
    files.append(root / "package_manifest.json")
    if zip_path.exists():
        raise FileExistsError(f"Archive already exists: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, root.name + "/" + path.relative_to(root).as_posix())
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.testzip() is None
        assert len(archive.infolist()) == len(files)
        for item in manifest["files"]:
            name = root.name + "/" + item["path"]
            with archive.open(name) as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"], name
    digest = sha256(zip_path)
    zip_path.with_suffix(".zip.sha256").write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    print(json.dumps({"zip": str(zip_path), "bytes": zip_path.stat().st_size,
          "files": len(files), "sha256": digest, "all_zipped_file_hashes_verified": True}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        finalize(args.output.resolve(), args.output.resolve().with_suffix(".zip"))
    else:
        assemble(args.repo.resolve(), args.output.resolve())
