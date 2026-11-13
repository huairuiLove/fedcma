"""Audit a finished suite and replay every best checkpoint."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import rankdata

from dataset import sha256, load_prepared
from evaluate_checkpoint import verify
from fed import save_json
from metrics import compute_metrics
from methods import method_components
from curriculum import curriculum_plan


def audit_online_weights(selected, prefix, draws, train, config, plan, actual_steps):
    """Independent CPU/SciPy reconstruction of every stored online weight."""
    point_losses, weights = selected[prefix + "_bce"], selected[prefix + "_weights"]
    assert point_losses.shape == weights.shape == draws.shape
    assert np.isfinite(point_losses).all() and np.isfinite(weights).all()
    alpha = plan.get("weight_strength", 0.)
    assert weights.min() >= 1 - alpha - 1e-6 and weights.max() <= 1 + alpha + 1e-6
    assert weights.min() > 0
    labels, masks = train["label"].numpy(), train["masks"].numpy()
    for step, start in enumerate(range(0, len(draws), config["batch_size"])):
        stop = start + config["batch_size"]
        ids, raw, observed = draws[start:stop], point_losses[start:stop], weights[start:stop]
        groups = labels[ids].astype(int) * 4 + masks[ids, 0].astype(int) + 2 * masks[ids, 1].astype(int)
        expected = np.ones(len(ids), dtype=np.float64)
        for group in np.unique(groups):
            which = groups == group
            n = int(which.sum())
            if n > 1:
                expected[which] += alpha * (2 * (rankdata(raw[which], method="average") - 1) / (n - 1) - 1)
            assert abs(float(observed[which].sum()) - n) < 2e-5
        np.testing.assert_allclose(observed, expected, rtol=0, atol=2e-7)
        record = actual_steps[step]
        assert record["curriculum_weight_strength"] == alpha
        assert abs(record["weight_sum"] - len(ids)) < 2e-5
        assert abs(record["task_unweighted"] - float(raw.mean())) < 2e-6
        assert abs(record["task"] - float((raw * observed).mean())) < 2e-6
        assert abs(record["weight_min"] - float(observed.min())) < 1e-7
        assert abs(record["weight_max"] - float(observed.max())) < 1e-7


def audit(directory):
    directory = Path(directory)
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    prepared = directory / "data" / "prepared.pt"
    if not prepared.exists():
        prepared = Path(protocol["runs"][0]["prepared"])
    manifest = json.loads((prepared.parent / "manifest.json").read_text(encoding="utf-8"))
    train = load_prepared(prepared, "cpu")[0]["train"]
    max_missing = float((1 - train["masks"].mean(1)).max())
    results = []
    for config in protocol["runs"]:
        run = directory / f"{config['method']}_s{config['seed']}"
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
        components = method_components(config["method"])
        if "components" in result:
            assert result["components"] == components
        history = json.loads((run / "history.json").read_text(encoding="utf-8"))
        assert len(history) == config["rounds"]
        assert result["best_round"] == max(history, key=lambda r: r["val"]["auprc"])["round"]
        assert result["data_sha256"] == manifest["prepared_sha256"]
        with (run / "steps.jsonl").open(encoding="utf-8") as stream:
            steps = [json.loads(line) for line in stream]
        assert len(steps) == sum(row["optimizer_steps"] for row in history)
        assert all(np.isfinite([row[k] for k in ("task", "align", "reg", "prox", "total", "grad_norm")]).all()
                   for row in steps)
        if not components["prior"]:
            assert all(row["align"] == 0 and row["reg"] == 0 for row in steps)
        if not components["proximal"]:
            assert all(row["prox"] == 0 for row in steps)
        for record in history:
            plan = curriculum_plan(SimpleNamespace(**config), record["round"] - 1,
                                   components["curriculum"], max_missing)
            if "curriculum" in record:
                assert record["curriculum"] == plan
            assert (run / "checkpoints" / f"round_{record['round']:02d}.pt").exists()
            with np.load(run / f"selection_round_{record['round']:02d}.npz") as selected:
                for k in range(config["num_clients"]):
                    full, kept = selected[f"client_{k}_all"], selected[f"client_{k}_selected"]
                    np.testing.assert_array_equal(full, np.flatnonzero(train["client"].numpy() == k))
                    assert len(kept) == record["selected"][k]
                    if plan["active"] and config.get("curriculum_schedule") != "balanced_weight":
                        scores = selected[f"client_{k}_scores"]
                        assert np.isfinite(scores).all()
                        np.testing.assert_array_equal(kept, full[scores <= record["rho"] + plan["tolerance"]])
                    else:
                        np.testing.assert_array_equal(full, kept)
                        assert f"client_{k}_scores" not in selected.files
                    if config.get("equal_training_budget", False):
                        for epoch in range(1, config["local_epochs"] + 1):
                            draws = selected[f"client_{k}_epoch_{epoch}_train"]
                            assert len(draws) == len(full)
                            unique, counts = np.unique(draws, return_counts=True)
                            np.testing.assert_array_equal(unique, np.sort(kept))
                            assert counts.max() - counts.min() <= 1
                            if config.get("curriculum_schedule") in ("stratified_order", "balanced_weight"):
                                np.testing.assert_array_equal(kept, full)
                                assert (counts == 1).all()
                                baseline = selected[f"client_{k}_epoch_{epoch}_random"]
                                np.testing.assert_array_equal(np.sort(baseline), np.sort(full))
                                # Position-wise matching implies exact equality for every batch,
                                # including the smaller final batch, for labels and mask patterns.
                                np.testing.assert_array_equal(train["label"].numpy()[draws], train["label"].numpy()[baseline])
                                np.testing.assert_array_equal(train["masks"].numpy()[draws], train["masks"].numpy()[baseline])
                                if not plan["active"] or config.get("curriculum_schedule") == "balanced_weight":
                                    np.testing.assert_array_equal(draws, baseline)
                            actual_steps = [s for s in steps if (s["round"], s["client"], s["epoch"]) ==
                                            (record["round"], k, epoch)]
                            expected_sizes = [min(config["batch_size"], len(full) - offset)
                                              for offset in range(0, len(full), config["batch_size"])]
                            assert [s["step"] for s in actual_steps] == list(range(len(expected_sizes)))
                            assert [s["n"] for s in actual_steps] == expected_sizes
                            if config.get("curriculum_schedule") == "balanced_weight":
                                audit_online_weights(selected, f"client_{k}_epoch_{epoch}", draws,
                                                     train, config, plan, actual_steps)
                if config.get("equal_training_budget", False):
                    assert record["examples_processed"] == len(train["label"]) * config["local_epochs"]
                    assert record["optimizer_steps"] == sum(
                        (n + config["batch_size"] - 1) // config["batch_size"] for n in record["client_sizes"]
                    ) * config["local_epochs"]
        if "training_budget" in result:
            assert result["training_budget"]["examples_processed"] == sum(s["n"] for s in steps)
            assert result["training_budget"]["optimizer_steps"] == len(steps)
        hashes = json.loads((run / "source_sha256.json").read_text(encoding="utf-8"))
        assert all(sha256(run / "source" / name) == value for name, value in hashes.items())
        replay = verify(run, prepared)
        if not config["validation_only"]:
            with (run / "test_predictions.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            assert [r["sample_id"] for r in rows] == manifest["splits"]["test"]["sample_ids"]
            recomputed = compute_metrics([int(r["label"]) for r in rows], [float(r["probability"]) for r in rows])
            for metric in ("auc", "auprc", "bce"):
                assert abs(recomputed[metric] - result["test_global"][metric]) < 1e-12
        results.append({**replay, "step_records": len(steps),
                        "training_examples": sum(s["n"] for s in steps),
                        "equal_budget_verified": config.get("equal_training_budget", False),
                        "every_sample_once_and_batch_strata_verified": config.get("curriculum_schedule") in ("stratified_order", "balanced_weight"),
                        "online_weights_and_exact_random_order_verified": config.get("curriculum_schedule") == "balanced_weight",
                        "checks_passed": True})
    report = {"all_passed": True, "n_runs": len(results), "runs": results}
    save_json(directory / "verification.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    audit(parser.parse_args().directory)
