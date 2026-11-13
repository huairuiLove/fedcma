"""Auditable FedAvg, FedProx and patent FedCMA on a single local GPU/CPU.

Only validation AP selects a checkpoint. Test evaluation happens once after
training. Per-step losses, sample selection and every round's state are saved.
"""
import copy
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from dataset import load_prepared, sha256
from metrics import compute_metrics
from model import FedCMAModel
from methods import method_components
from curriculum import curriculum_plan, training_order, stratified_curriculum_order, balanced_hardness_weights


def save_json(path, content):
    Path(path).write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def seed_everything(seed, threads=4):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def take(data, indices):
    return {key: values[indices] for key, values in data.items()}


def difficulty_score(losses, masks, xi):
    """Patent C_i = current BCE / batch maximum BCE, P_i = missing fraction."""
    complexity = losses / losses.max().clamp_min(1e-12)
    missingness = 1 - masks.mean(dim=1)
    return xi * missingness + (1 - xi) * complexity


@torch.no_grad()
def curriculum_selection(model, data, indices, prior, cfg, rho, tolerance=0.):
    model.eval()
    scores = []
    use_client_normalizer = getattr(cfg, "curriculum_schedule", "legacy") == "stratified_order"
    for chunk in indices.split(cfg.batch_size):
        batch = take(data, chunk)
        logits, _ = model(batch, prior)
        losses = F.binary_cross_entropy_with_logits(logits, batch["label"], reduction="none")
        scores.append(losses if use_client_normalizer else difficulty_score(losses, batch["masks"], cfg.xi))
    score = torch.cat(scores)
    if use_client_normalizer:
        score = difficulty_score(score, data["masks"][indices], cfg.xi)
        return indices, score
    return indices[score <= rho + tolerance], score


def sample_counts(data, indices):
    labels = data["label"][indices]
    incomplete = data["masks"][indices].sum(1) < data["masks"].shape[1]
    positive, missing, missing_positive = torch.stack(
        [labels.sum(), incomplete.sum(), (labels * incomplete).sum()]).cpu().tolist()
    return {"n": len(indices), "positive": int(positive), "incomplete": int(missing),
            "incomplete_positive": int(missing_positive)}


def aggregate(states, counts):
    weights = np.asarray(counts, dtype=float)
    weights /= weights.sum()
    return {key: sum(float(w) * state[key] for w, state in zip(weights, states))
            for key in states[0]}


def proximal_penalty(model, initial):
    return sum((p - initial[name]).square().sum()
               for name, p in model.named_parameters() if p.requires_grad)


@torch.no_grad()
def predict(model, rows, prior, batch_size=256):
    model.eval()
    scores = []
    for indices in torch.arange(len(rows["label"]), device=rows["label"].device).split(batch_size):
        logits, _ = model(take(rows, indices), prior)
        scores.append(torch.sigmoid(logits).cpu())
    return torch.cat(scores).numpy()


def evaluate(model, rows, prior):
    scores = predict(model, rows, prior)
    return compute_metrics(rows["label"].cpu().numpy(), scores), scores


def export_predictions(path, rows, ids, scores):
    y, masks, clients = (rows[k].cpu().numpy() for k in ("label", "masks", "client"))
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "client", "label", "ts_present", "text_present", "probability"])
        for sid, c, label, mask, probability in zip(ids, clients, y, masks, scores):
            writer.writerow([sid, int(c), int(label), int(mask[0]), int(mask[1]), float(probability)])


def grouped_metrics(rows, scores):
    y, masks = rows["label"].cpu().numpy(), rows["masks"].cpu().numpy()
    groups = {"complete": masks.sum(1) == 2, "incomplete": masks.sum(1) == 1,
              "ts_missing": masks[:, 0] == 0, "text_missing": masks[:, 1] == 0}
    return {name: compute_metrics(y[sel], scores[sel]) for name, sel in groups.items() if sel.any()}


def runtime_info(cfg):
    import importlib.metadata
    info = {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda, "device": cfg.device,
            "deterministic_algorithms": True, "precision": "float32", "tf32": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("torch", "numpy", "scipy", "scikit-learn", "matplotlib")}}
    if cfg.device.startswith("cuda"):
        prop = torch.cuda.get_device_properties(cfg.device)
        info.update(gpu=prop.name, gpu_memory_bytes=prop.total_memory)
    try:
        import psutil
        info.update(ram_bytes=psutil.virtual_memory().total, logical_cpus=psutil.cpu_count())
    except ImportError:
        pass
    return info


def run_fed(cfg):
    output = Path(cfg.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    seed_everything(cfg.seed, cfg.threads)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(cfg).items()}
    save_json(output / "config.json", config)
    save_json(output / "environment.json", runtime_info(cfg))
    root = Path(__file__).resolve().parent
    source = output / "source"
    source.mkdir()
    for file in root.glob("*.py"):
        shutil.copy2(file, source / file.name)
    save_json(output / "source_sha256.json", {p.name: sha256(p) for p in source.glob("*.py")})
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (output / "environment-freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    data, manifest = load_prepared(cfg.prepared, cfg.device)
    if cfg.num_clients != manifest["settings"]["num_clients"]:
        raise ValueError("num_clients differs from prepared client allocation")
    save_json(output / "data_reference.json", {"path": str(cfg.prepared.resolve()),
                                               "sha256": manifest["prepared_sha256"]})

    def log(message):
        line = f"{datetime.now(timezone.utc).isoformat()} {message}"
        print(line, flush=True)
        with (output / "train.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    components = method_components(cfg.method)
    use_prior = components["prior"]
    use_curriculum = components["curriculum"]
    global_model = FedCMAModel(cfg).to(cfg.device)
    model = FedCMAModel(cfg).to(cfg.device)
    priors = torch.randn(cfg.num_clients, 2, cfg.embed_dim, device=cfg.device) * 0.02
    train = data["train"]
    client_indices = [torch.where(train["client"] == k)[0] for k in range(cfg.num_clients)]
    client_sizes = [len(indices) for indices in client_indices]
    max_missing_fraction = float((1 - train["masks"].mean(1)).max().item())
    equal_budget = getattr(cfg, "equal_training_budget", False)
    ordering_curriculum = getattr(cfg, "curriculum_schedule", "legacy") == "stratified_order"
    weighting_curriculum = getattr(cfg, "curriculum_schedule", "legacy") == "balanced_weight"
    record_draws = equal_budget or getattr(cfg, "curriculum_schedule", "legacy") != "legacy"
    history, best_value, best_round = [], -1., -1
    started = time.perf_counter()
    n_parameters = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    log(f"method={cfg.method} seed={cfg.seed} train={len(train['label'])} parameters={n_parameters}")
    if cfg.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with (output / "steps.jsonl").open("w", encoding="utf-8") as step_file:
        for round_index in range(cfg.rounds):
            fraction = round_index / max(cfg.rounds - 1, 1)
            plan = curriculum_plan(cfg, round_index, use_curriculum, max_missing_fraction)
            rho = plan["rho"]
            delta = cfg.delta0 + (cfg.delta_max - cfg.delta0) * fraction
            local_states, local_priors, round_records, selections = [], [], [], {}
            selection_seconds, selection_examples, selection_diagnostics = 0., 0, []
            for k, indices in enumerate(client_indices):
                torch.manual_seed(cfg.seed * 100000 + round_index * 1000 + k)
                model.load_state_dict(global_model.state_dict())
                prior = priors[k].detach().clone().requires_grad_(use_prior)
                selection_started = time.perf_counter()
                if plan["active"] and not weighting_curriculum:
                    selected, scores = curriculum_selection(
                        model, train, indices, prior if use_prior else None, cfg, rho, plan["tolerance"])
                    selections[f"client_{k}_scores"] = scores.cpu().numpy()
                    selection_examples += len(indices)
                else:
                    selected = indices
                selection_seconds += time.perf_counter() - selection_started
                if len(selected) == 0:
                    raise ValueError(f"Empty curriculum pool: round={round_index+1} client={k} rho={rho}")
                selections[f"client_{k}_all"] = indices.cpu().numpy()
                selections[f"client_{k}_selected"] = selected.cpu().numpy()
                if record_draws:
                    diagnostic = {"client": k, "full": sample_counts(train, indices),
                                  "selected": sample_counts(train, selected)}
                    if plan["active"] and not weighting_curriculum:
                        diagnostic["score_quantiles"] = dict(zip(
                            ("min", "q25", "q50", "q75", "q90", "max"),
                            np.quantile(selections[f"client_{k}_scores"], [0, .25, .5, .75, .9, 1]).tolist()))
                    selection_diagnostics.append(diagnostic)
                params = [p for p in model.parameters() if p.requires_grad] + ([prior] if use_prior else [])
                optimizer = torch.optim.Adam(params, lr=cfg.lr)
                initial = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
                model.train()
                for epoch in range(cfg.local_epochs):
                    generator = torch.Generator(device=cfg.device).manual_seed(
                        cfg.seed * 100000 + round_index * 1000 + k * 10 + epoch)
                    shuffled = training_order(selected, len(indices) if equal_budget else len(selected), generator)
                    if ordering_curriculum or weighting_curriculum:
                        selections[f"client_{k}_epoch_{epoch+1}_random"] = shuffled.cpu().numpy()
                        if plan["active"] and ordering_curriculum:
                            shuffled = stratified_curriculum_order(
                                shuffled, indices, scores, train["label"], train["masks"], plan["order_strength"])
                    if record_draws:
                        selections[f"client_{k}_epoch_{epoch+1}_train"] = shuffled.cpu().numpy()
                    epoch_weights = []
                    for step, chunk in enumerate(shuffled.split(cfg.batch_size)):
                        if not len(chunk):
                            continue
                        batch = take(train, chunk)
                        logits, real = model(batch, prior if use_prior else None)
                        batch_weights = None
                        if weighting_curriculum:
                            with torch.no_grad():
                                point_bce = F.binary_cross_entropy_with_logits(logits, batch["label"], reduction="none")
                                batch_weights = balanced_hardness_weights(
                                    point_bce, batch["label"], batch["masks"], plan.get("weight_strength", 0.))
                                epoch_weights.append(torch.stack([point_bce, batch_weights]).cpu().numpy())
                        loss, parts = model.loss(logits, batch, real, prior if use_prior else None,
                                                 sample_weights=batch_weights if weighting_curriculum and plan["active"] else None)
                        prox = cfg.mu / 2 * proximal_penalty(model, initial) if components["proximal"] else loss.new_zeros(())
                        total = loss + prox
                        optimizer.zero_grad(set_to_none=True)
                        if not torch.isfinite(total):
                            raise FloatingPointError(f"Non-finite loss round={round_index} client={k}")
                        total.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip, error_if_nonfinite=True)
                        optimizer.step()
                        values = torch.stack([parts["task"], parts["align"], parts["reg"], prox, total, grad_norm]).detach().cpu().tolist()
                        rec = dict(zip(("task", "align", "reg", "prox", "total", "grad_norm"), values))
                        rec.update(round=round_index + 1, client=k, epoch=epoch + 1, step=step,
                                   n=len(chunk), lr=cfg.lr, elapsed_seconds=time.perf_counter() - started)
                        if weighting_curriculum:
                            raw, minimum, maximum, mass = torch.stack([
                                parts["task_unweighted"], batch_weights.min(), batch_weights.max(), batch_weights.sum()
                            ]).detach().cpu().tolist()
                            rec.update(task_unweighted=raw, weight_min=minimum, weight_max=maximum,
                                       weight_sum=mass, curriculum_weight_strength=plan.get("weight_strength", 0.))
                        step_file.write(json.dumps(rec, allow_nan=False) + "\n")
                        round_records.append(rec)
                    if weighting_curriculum:
                        saved = np.concatenate(epoch_weights, axis=1)
                        selections[f"client_{k}_epoch_{epoch+1}_bce"] = saved[0]
                        selections[f"client_{k}_epoch_{epoch+1}_weights"] = saved[1]
                step_file.flush()
                local_states.append({n: p.detach().clone() for n, p in model.state_dict().items()})
                local_priors.append(prior.detach().clone())
                log(f"round={round_index+1} client={k} selected={len(selected)}/{len(indices)}")
            # Patent weights use original local n_k, not the curriculum-selected n_k.
            global_model.load_state_dict(aggregate(local_states, client_sizes))
            weights = torch.tensor(client_sizes, dtype=torch.float32, device=cfg.device)
            weights = weights / weights.sum()
            pbar = sum(weights[k] * local_priors[k] for k in range(cfg.num_clients))
            if use_prior:
                priors = torch.stack([(1 - delta) * p + delta * pbar for p in local_priors])
            ref = pbar if use_prior else None
            val, val_scores = evaluate(global_model, data["val"], ref)
            count = sum(r["n"] for r in round_records)
            loss_means = {key: sum(r[key] * r["n"] for r in round_records) / max(count, 1)
                          for key in ("task", "align", "reg", "prox", "total")}
            if weighting_curriculum:
                loss_means["task_unweighted"] = sum(r["task_unweighted"] * r["n"] for r in round_records) / max(count, 1)
            rec = {"round": round_index + 1, "rho": rho, "delta": delta,
                   "curriculum": plan, "equal_training_budget": equal_budget,
                   "selection_examples_scored": selection_examples, "selection_seconds": selection_seconds,
                   "selected": [len(selections[f"client_{k}_selected"]) for k in range(cfg.num_clients)],
                   "client_sizes": client_sizes, "examples_processed": count,
                   "optimizer_steps": len(round_records), "train_loss": loss_means, "val": val,
                   "elapsed_seconds": time.perf_counter() - started}
            if record_draws:
                rec["selection_diagnostics"] = selection_diagnostics
            if equal_budget:
                assert count == sum(client_sizes) * cfg.local_epochs
                assert len(round_records) == sum((n + cfg.batch_size - 1) // cfg.batch_size
                                                for n in client_sizes) * cfg.local_epochs
            history.append(rec)
            save_json(output / "history.json", history)
            np.savez_compressed(output / f"selection_round_{round_index+1:02d}.npz", **selections)
            export_predictions(output / f"val_round_{round_index+1:02d}.csv", data["val"],
                               manifest["splits"]["val"]["sample_ids"], val_scores)
            checkpoint = {"round": round_index + 1, "config": config,
                          "model": {n: v.cpu() for n, v in global_model.state_dict().items()},
                          "priors": priors.cpu(), "reference_prior": pbar.cpu(),
                          "validation": val, "prepared_sha256": manifest["prepared_sha256"],
                          "torch_rng": torch.get_rng_state(),
                          "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
            checkpoint_path = output / "checkpoints" / f"round_{round_index+1:02d}.pt"
            torch.save(checkpoint, checkpoint_path)
            if val["auprc"] > best_value:
                best_value, best_round = val["auprc"], round_index + 1
                shutil.copy2(checkpoint_path, output / "best.pt")
            log(f"round={round_index+1} loss={loss_means['total']:.5f} val_auc={val['auc']:.5f} "
                f"val_ap={val['auprc']:.5f} elapsed={rec['elapsed_seconds']:.1f}s")

    best = torch.load(output / "best.pt", map_location=cfg.device, weights_only=True)
    global_model.load_state_dict(best["model"])
    ref = best["reference_prior"] if use_prior else None
    result = {"method": cfg.method, "components": components, "seed": cfg.seed, "best_round": best_round,
              "selection_metric": "global_validation_average_precision",
              "best_validation": best["validation"], "trainable_parameters": n_parameters,
              "validation_only": cfg.validation_only, "data_sha256": manifest["prepared_sha256"]}
    result["training_budget"] = {
        "examples_processed": sum(r["examples_processed"] for r in history),
        "optimizer_steps": sum(r["optimizer_steps"] for r in history),
        "selection_examples_scored": sum(r["selection_examples_scored"] for r in history),
        "selection_seconds": sum(r["selection_seconds"] for r in history),
        "equal_training_budget": equal_budget}
    if not cfg.validation_only:
        metrics, scores = evaluate(global_model, data["test"], ref)
        result["test_global"] = metrics
        result["test_subgroups"] = grouped_metrics(data["test"], scores)
        export_predictions(output / "test_predictions.csv", data["test"], manifest["splits"]["test"]["sample_ids"], scores)
        # Secondary personalized results use POOLED predictions, not mean client AUC.
        personalized = scores.copy()
        if use_prior:
            for k in range(cfg.num_clients):
                indices = torch.where(data["test"]["client"] == k)[0]
                if len(indices):
                    personalized[indices.cpu().numpy()] = predict(global_model, take(data["test"], indices), best["priors"][k])
        result["test_personalized_pooled"] = compute_metrics(data["test"]["label"].cpu().numpy(), personalized)
        export_predictions(output / "test_personalized_predictions.csv", data["test"],
                           manifest["splits"]["test"]["sample_ids"], personalized)
    result["elapsed_seconds"] = time.perf_counter() - started
    result["peak_cuda_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20 if cfg.device.startswith("cuda") else None
    save_json(output / "result.json", result)
    log(f"COMPLETED {json.dumps(result, allow_nan=False)}")
    return result
