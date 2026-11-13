"""Rebuild frozen IHM tensors from the included processed input files."""
import argparse
import json
from pathlib import Path
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.package.resolve()
    if args.output.exists():
        raise FileExistsError("Use a new directory; the archived data will not be overwritten")
    archive = root / "records/experiments/batch_03_ihm_20261113/data"
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(root / "code"))
    import torch
    from dataset import prepare_data, load_prepared
    rebuilt = prepare_data(root / "input_data", args.output, **manifest["settings"])
    original, _ = load_prepared(archive / "prepared.pt", "cpu")
    actual, _ = load_prepared(args.output / "prepared.pt", "cpu")
    for split, fields in original.items():
        for name, tensor in fields.items():
            assert torch.equal(tensor, actual[split][name]), (split, name)
    assert rebuilt["splits"] == manifest["splits"]
    (args.output / "rebuild_verification.json").write_text(json.dumps({
        "all_passed": True, "all_tensors_identical": True, "splits_and_masks_identical": True,
        "archive_prepared_sha256": manifest["prepared_sha256"],
        "rebuilt_prepared_sha256": rebuilt["prepared_sha256"]}, indent=2), encoding="utf-8")
    print("All reconstructed tensors, labels, splits, masks and client assignments match batch 3.")
