"""Federated multimodal dataset for FedCMA.

Loads the p2x pickles + precomputed note embeddings, builds
- regularized hourly grid reg_ts (window x 2*17) by forward-filling the
  normalized irregular observations (backbone F_impute),
- fixed sample-level missingness masks I^(m) ~ Bernoulli(1-q) per modality,
  resampled until at least one modality is present (seed fixed across methods,
  identical at train and test time, as in the paper),
- difficulty scores S_i = xi * P_i + (1-xi) * C_i with P_i = 1 - (I_ts+I_txt)/2
  and C_i = client-normalized observation counts,
- Dirichlet(alpha) client partitions of the train/val/test cohorts.
"""
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

NUM_NOTES = 5  # backbone default: last 5 notes


def f_impute(irg, mask, tt, window):
    """Hourly forward-fill grid of (value|mask) channels, [window, 2*d]."""
    d = irg.shape[1]
    out = np.zeros((int(window), 2 * d), dtype=np.float32)
    for x, t, m in zip(irg, tt, mask):
        row = int(t)
        if row >= window:
            continue
        for j in range(d):
            if m[j] == 1:
                out[row, j] = x[j]
                out[row, d + j] = 1.0
            elif row > 0 and out[row - 1, d + j] == 1.0:
                out[row, j] = out[row - 1, j]
                out[row, d + j] = 1.0
    return out


def make_masks(n, q, rng):
    """Bernoulli(1-q) per modality; resample while both are 0."""
    m = rng.random((n, 2)) > q
    both = m.sum(axis=1) == 0
    while both.any():
        idx = np.where(both)[0]
        m[idx] = rng.random((len(idx), 2)) > q
        both = m.sum(axis=1) == 0
    return m.astype(bool)


def dirichlet_partition(n, k, alpha, rng):
    """Sample counts per client ~ Dir(alpha), then assign a random permutation.
    Guarantees >=1 sample per client when n >= k."""
    parts = np.sort(rng.permutation(n))
    cuts = [0]
    for c in rng.dirichlet([alpha] * k):
        cuts.append(min(n, cuts[-1] + int(round(c * n))))
    cuts[-1] = n
    sizes = [cuts[i + 1] - cuts[i] for i in range(k)]
    if n >= k:  # steal from the largest client to feed any empty client
        for i in range(k):
            if sizes[i] == 0:
                big = int(np.argmax(sizes))
                if sizes[big] <= 1:
                    break
                sizes[big] -= 1
                sizes[i] += 1
        cuts = [0]
        for s in sizes:
            cuts.append(cuts[-1] + s)
    assign = np.zeros(n, dtype=np.int64)
    for cid in range(k):
        assign[parts[cuts[cid]:cuts[cid + 1]]] = cid
    return assign


class FLDataset(Dataset):
    def __init__(self, task, split, data_dir, cfg):
        with open(f"{data_dir}/{task}/{split}_p2x.pkl", "rb") as f:
            raw = pickle.load(f)
        with open(f"{data_dir}/{task}/{split}_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        by_name = {m["name"]: m["note_ids"] for m in meta}
        emb = np.load(f"{data_dir}/{task}/notes_{split}.npz")["emb"].astype(np.float32)
        self.window = cfg.tt_max
        self.task = task

        self.samples = []
        for s in raw:
            reg = f_impute(s["irg_ts"], s["irg_ts_mask"], s["ts_tt"], self.window)
            note_ids = by_name[s["name"]][-NUM_NOTES:]
            ne = emb[note_ids] if note_ids else np.zeros((0, 768), np.float32)
            # backbone normalizes note time as 1 - time_to_end / tt_max
            nte = s["text_time_to_end"][-NUM_NOTES:]
            nt = 1.0 - nte / self.window
            self.samples.append({
                "irg": torch.from_numpy(s["irg_ts"]),
                "mask": torch.from_numpy(s["irg_ts_mask"]),
                "tt": torch.from_numpy(s["ts_tt"] / self.window),
                "reg": torch.from_numpy(reg),
                "note_emb": torch.from_numpy(ne),
                "note_time": torch.from_numpy(nt.astype(np.float32)),
                "label": torch.tensor(s["label"]),
                "n_ts": int(s["irg_ts_mask"].sum()),
                "n_txt": len(note_ids),
            })

        rng = np.random.default_rng(cfg.mask_seed)
        self.masks = make_masks(len(self.samples), cfg.q, rng)

        # difficulty (needs the client assignment first for C_i)
        self.client = dirichlet_partition(len(self.samples), cfg.num_clients,
                                          cfg.dir_alpha,
                                          np.random.default_rng(cfg.partition_seed))

    def __len__(self):
        return len(self.samples)

    def difficulty(self, xi=0.5):
        """S_i = xi*P_i + (1-xi)*C_i; C_i client-normalized counts (tex Eq.9-11)."""
        P = 1.0 - self.masks.sum(axis=1) / 2.0
        C = np.zeros(len(self.samples))
        for k in np.unique(self.client):
            sel = self.client == k
            n_ts = np.array([self.samples[i]["n_ts"] for i in np.where(sel)[0]], float)
            n_txt = np.array([self.samples[i]["n_txt"] for i in np.where(sel)[0]], float)
            c = 0.5 * (n_ts / max(n_ts.max(), 1) + n_txt / max(n_txt.max(), 1))
            C[sel] = np.clip(c, 0, 1)
        return xi * P + (1 - xi) * C

    def __getitem__(self, i):
        s = self.samples[i]
        n = len(s["note_emb"])
        if n < NUM_NOTES:  # pad notes
            pad = NUM_NOTES - n
            note_emb = torch.cat([s["note_emb"], torch.zeros(pad, 768)], 0)
            note_time = torch.cat([s["note_time"], torch.zeros(pad)])
            note_mask = torch.cat([torch.ones(n), torch.zeros(pad)])
        else:
            note_emb = s["note_emb"]
            note_time = s["note_time"]
            note_mask = torch.ones(n)
        i_ts, i_txt = self.masks[i]
        return {
            "irg": s["irg"], "irg_mask": s["mask"], "tt": s["tt"], "reg": s["reg"],
            "note_emb": note_emb, "note_time": note_time,
            "note_mask": note_mask.long(),
            "i_ts": torch.tensor(float(i_ts)), "i_txt": torch.tensor(float(i_txt)),
            "complete": torch.tensor(float(i_ts and i_txt)),
            "label": s["label"], "idx": torch.tensor(i),
        }


def collate(batch):
    out = {}
    for k in ["irg", "irg_mask", "tt", "note_emb", "note_time", "note_mask"]:
        out[k] = torch.nn.utils.rnn.pad_sequence([b[k] for b in batch],
                                                 batch_first=True)
    for k in ["reg", "i_ts", "i_txt", "complete", "label", "idx"]:
        out[k] = torch.stack([b[k] for b in batch])
    return out
