"""Patient-grouped IHM data with training-only preprocessing and audit manifest."""
from pathlib import Path
import hashlib
import json
import pickle

import numpy as np
import torch
from sklearn.model_selection import train_test_split


class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {("numpy", "ndarray"), ("numpy", "dtype")}
        allowed.update((m, n) for m in ("numpy.core.multiarray", "numpy._core.multiarray")
                       for n in ("_reconstruct", "scalar"))
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unexpected object: {module}.{name}")
        return super().find_class(module, name)


def read_arrays(path):
    with open(path, "rb") as stream:
        return ArrayUnpickler(stream).load()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def private_id(value):
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def fit_normalizer(samples):
    """Fit winsorization and standardization on selected TRAIN observations."""
    stats = {key: [] for key in ("lower", "upper", "mean", "std")}
    for j in range(17):
        values = np.concatenate([s["irg_ts"][s["irg_ts_mask"][:, j] == 1, j]
                                 for s in samples]).astype(np.float64)
        if not len(values):
            values = np.zeros(1)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite training observations")
        lo, hi = np.quantile(values, [0.005, 0.995])
        clipped = np.clip(values, lo, hi)
        for key, value in zip(stats, (lo, hi, clipped.mean(), max(clipped.std(), 1e-6))):
            stats[key].append(float(value))
    return stats


def f_impute(values, mask, times, window=48):
    """Carry values through empty hours; keep the actual hourly observation mask."""
    d = values.shape[1]
    hourly = np.zeros((window, d), dtype=np.float32)
    observed = np.zeros_like(hourly)
    for i in np.argsort(times, kind="stable"):
        row = int(np.floor(times[i]))
        if not 0 <= row < window:
            continue
        present = mask[i].astype(bool)
        hourly[row, present] = values[i, present]
        observed[row, present] = 1
    previous = np.zeros(d, dtype=np.float32)
    for row in range(window):
        present = observed[row].astype(bool)
        previous[present] = hourly[row, present]
        hourly[row] = previous
    return np.concatenate([hourly, observed], axis=1)


def make_masks(n, missing_rate, rng):
    """Exactly round(rate*N) incomplete samples, each missing ONE modality.

    rate=0.5 means 50% incomplete samples and 25% missing modality entries.
    """
    if not 0 <= missing_rate <= 1:
        raise ValueError("missing_rate must be in [0, 1]")
    masks = np.ones((n, 2), dtype=np.float32)
    selected = rng.permutation(n)[:round(n * missing_rate)]
    masks[selected, rng.integers(0, 2, len(selected))] = 0
    return masks


def partition_patients(patient_ids, labels, k, alpha, rng):
    """Class-conditional Dirichlet TRAIN allocation; one patient, one client."""
    patients = np.unique(patient_ids)
    if len(patients) < k:
        raise ValueError("Fewer training patients than clients")
    grouped = {p: np.flatnonzero(patient_ids == p) for p in patients}
    patient_label = {p: int(np.max(labels[grouped[p]])) for p in patients}
    for _ in range(1000):
        mapping = {}
        for label in (0, 1):
            group = np.array([p for p in patients if patient_label[p] == label])
            rng.shuffle(group)
            sizes = rng.multinomial(len(group), rng.dirichlet(np.full(k, alpha)))
            for cid, part in enumerate(np.split(group, np.cumsum(sizes)[:-1])):
                mapping.update({p: cid for p in part})
        assignment = np.array([mapping[p] for p in patient_ids], dtype=np.int64)
        if np.bincount(assignment, minlength=k).min() >= min(20, len(labels) // (2 * k)):
            return assignment
    raise ValueError("Could not create usable client allocation")


def prepare_data(data_dir, output_dir, train_size=3072, subset_seed=2026,
                 mask_seed=42, partition_seed=42, missing_rate=0.5,
                 num_clients=5, dir_alpha=0.5):
    root, destination = Path(data_dir) / "ihm", Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "prepared.pt").exists():
        raise FileExistsError("Prepared data exists; use a new experiment directory")
    raw = {split: read_arrays(root / f"{split}_p2x.pkl") for split in ("train", "val", "test")}
    patients = {split: {s["name"].split("_")[0] for s in samples} for split, samples in raw.items()}
    overlaps = {f"{a}_{b}": len(patients[a] & patients[b])
                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    if any(overlaps.values()):
        raise ValueError(f"Patient leakage: {overlaps}")
    source_counts = {split: len(samples) for split, samples in raw.items()}
    train_idx = np.arange(len(raw["train"]))
    if train_size and train_size < len(train_idx):
        train_idx, _ = train_test_split(train_idx, train_size=train_size,
                                       random_state=subset_seed,
                                       stratify=[s["label"] for s in raw["train"]])
        train_idx.sort()
    raw["train"] = [raw["train"][int(i)] for i in train_idx]
    normalizer = fit_normalizer(raw["train"])
    low, high, mean, std = (np.array(normalizer[key], np.float32)
                            for key in ("lower", "upper", "mean", "std"))
    cached, metadata, text_features = {}, {}, {}
    for split_index, (split, samples) in enumerate(raw.items()):
        by_name = {s["name"]: s["note_ids"] for s in read_arrays(root / f"{split}_meta.pkl")}
        with np.load(root / f"notes_{split}.npz") as archive:
            emb = archive["emb"].astype(np.float32)
        grids, summaries, texts = [], [], []
        for s in samples:
            mask, times = s["irg_ts_mask"], s["ts_tt"]
            vals = (np.clip(s["irg_ts"], low, high) - mean) / std
            vals = np.where(mask == 1, vals, 0).astype(np.float32)
            grids.append(f_impute(vals, mask, times))
            channel_features = []
            for j in range(17):
                obs = vals[mask[:, j] == 1, j]
                channel_features.extend([float(obs.mean()), float(obs.std()), float(obs.min()),
                                         float(obs.max()), float(obs[-1]), float((mask[:, j] == 1).mean())]
                                        if len(obs) else [0.] * 6)
            summaries.append(channel_features)
            note_ids = by_name[s["name"]][-5:]
            if not note_ids:
                raise ValueError("The base cohort must have both modalities")
            texts.append(emb[note_ids].mean(axis=0))
        text_features[split] = np.stack(texts)
        labels = np.array([s["label"] for s in samples], dtype=np.float32)
        ids = np.array([s["name"].split("_")[0] for s in samples])
        if split == "train":
            clients = partition_patients(ids, labels, num_clients, dir_alpha,
                                         np.random.default_rng(partition_seed))
            fractions = np.bincount(clients, minlength=num_clients) / len(clients)
        else:
            # Evaluation client allocation uses patient ID and TRAIN client sizes,
            # never evaluation labels. Global pooled metrics are the primary result.
            mapping = {}
            for patient in np.unique(ids):
                digest = hashlib.sha256(f"{partition_seed}:{patient}".encode()).digest()
                u = int.from_bytes(digest[:8], "big") / 2**64
                mapping[patient] = min(num_clients - 1, int(np.searchsorted(np.cumsum(fractions), u)))
            clients = np.array([mapping[p] for p in ids], dtype=np.int64)
        masks = make_masks(len(samples), missing_rate,
                           np.random.default_rng(mask_seed + 1009 * split_index))
        cached[split] = {"ts": torch.from_numpy(np.stack(grids)),
                         "summary": torch.tensor(summaries, dtype=torch.float32),
                         "label": torch.from_numpy(labels), "masks": torch.from_numpy(masks),
                         "client": torch.from_numpy(clients)}
        metadata[split] = {
            "source_indices": train_idx.tolist() if split == "train" else list(range(len(samples))),
            "sample_ids": [private_id(s["name"]) for s in samples],
            "patient_ids": [private_id(p) for p in ids], "n": len(samples),
            "n_patients": len(np.unique(ids)), "positives": int(labels.sum()),
            "positive_rate": float(labels.mean()),
            "incomplete_sample_rate": float(np.mean(masks.sum(axis=1) == 1)),
            "missing_entry_rate": float(1 - masks.mean()),
            "mask_counts": {"complete": int((masks.sum(axis=1) == 2).sum()),
                            "ts_missing": int((masks[:, 0] == 0).sum()),
                            "text_missing": int((masks[:, 1] == 0).sum())},
            "clients": [{"client": k, "n": int((clients == k).sum()),
                         "positives": int(labels[clients == k].sum())} for k in range(num_clients)]}
    text_mean = text_features["train"].mean(axis=0)
    text_std = np.maximum(text_features["train"].std(axis=0), 1e-5)
    for split in cached:
        cached[split]["text"] = torch.from_numpy(np.clip(
            (text_features[split] - text_mean) / text_std, -5, 5).astype(np.float32))
        if not all(torch.isfinite(v).all() for v in cached[split].values()):
            raise ValueError(f"Non-finite features in {split}")
    torch.save(cached, destination / "prepared.pt")
    inputs = [f"{s}_{kind}.pkl" for s in raw for kind in ("p2x", "meta")]
    inputs += [f"notes_{s}.npz" for s in raw] + ["stats.json"]
    manifest = {"schema_version": 1, "task": "ihm", "source_counts": source_counts,
                "patient_overlap": overlaps, "splits": metadata,
                "settings": {"train_size": train_size, "subset_seed": subset_seed,
                             "mask_seed": mask_seed, "partition_seed": partition_seed,
                             "missing_rate": missing_rate, "num_clients": num_clients,
                             "dir_alpha": dir_alpha},
                "normalizer": normalizer, "text_mean": text_mean.tolist(),
                "text_std": text_std.tolist(),
                "input_sha256": {name: sha256(root / name) for name in inputs},
                "prepared_sha256": sha256(destination / "prepared.pt")}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_prepared(path, device):
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if sha256(path) != manifest["prepared_sha256"]:
        raise ValueError("Prepared data hash does not match manifest")
    data = torch.load(path, map_location="cpu", weights_only=True)
    return {s: {k: v.to(device) for k, v in rows.items()} for s, rows in data.items()}, manifest
