"""Step 2: build per-sample tensors (backbone "p2x" format) for FedCMA.

For each cohort episode: irregular ts matrix (l_ts x 17, categoricals coded),
mask, time points (hours, capped at the task window), note texts, note
time-to-window-end, labels. Irregular ts values are z-scored per channel with
statistics computed on the TRAIN split (masked observations only).

Outputs data/{task}/{split}_p2x.pkl: list of dicts with keys
  name, irg_ts (float32 [l,17]), irg_ts_mask (float32), ts_tt (float32 [l]),
  text_data (list[str]), text_time_to_end (float32 [n]), label (int | [25] f32)
and data/{task}/stats.json (channel mean/std).
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MIMIC_ROOT = "/Volumes/TiPlus7100s/mimic"
EXTRACT = f"{MIMIC_ROOT}/mimic-iv-extracted"
DATA = os.path.join(HERE, "data")

with open(os.path.join(HERE, "configs", "discretizer_config.json")) as f:
    DISC = json.load(f)
with open(os.path.join(HERE, "configs", "channel_info.json")) as f:
    CHINFO = json.load(f)

CHANNELS = DISC["id_to_channel"]          # 17 channels, canonical order
IS_CAT = DISC["is_categorical_channel"]

TASKS = {"ihm": {"dir": "in-hospital-mortality", "window": 48.0},
         "pheno": {"dir": "phenotyping", "window": 24.0}}


def clean_text(raw: str) -> str:
    return " ".join(" ".join(raw.splitlines()).split())


def load_notes(row):
    offsets = [float(x) for x in str(row["note_offsets_h"]).split(";")]
    paths = str(row["report_paths"]).split(";")
    notes, offs = [], []
    for off, p in zip(offsets, paths):
        with open(p, errors="ignore") as f:
            t = clean_text(f.read())
        if t:
            notes.append(t)
            offs.append(off)
    return notes, np.array(offs, dtype=np.float32)


def load_episode(split_dir, stay, window):
    df = pd.read_csv(os.path.join(split_dir, stay))
    df = df[df["Hours"] < window]
    tt = df["Hours"].to_numpy(dtype=np.float32)
    feats = np.zeros((len(df), len(CHANNELS)), dtype=np.float32)
    mask = np.zeros((len(df), len(CHANNELS)), dtype=np.float32)
    for j, ch in enumerate(CHANNELS):
        col = df[ch]
        notna = col.notna().to_numpy()
        mask[:, j] = notna.astype(np.float32)
        if not notna.any():
            continue
        if IS_CAT[ch]:
            vmap = CHINFO[ch]["values"]
            vals = col[notna].astype(str).map(lambda v: vmap[v])
            feats[notna, j] = vals.to_numpy(dtype=np.float32)
        else:
            feats[notna, j] = col[notna].astype(float).to_numpy(dtype=np.float32)
    return tt, feats, mask


def build(task):
    cfg = TASKS[task]
    out_dir = os.path.join(DATA, task)
    splits = {}
    for split in ["train", "val", "test"]:
        cohort = pd.read_csv(os.path.join(out_dir, f"cohort_{split}.csv"))
        split_dir = os.path.join(EXTRACT, cfg["dir"], "train" if split != "test" else "test")
        if task == "ihm":
            labels = cohort["y_true"].to_numpy(dtype=np.int64)
        else:
            label_cols = [c for c in cohort.columns
                          if c not in ("stay", "period_length", "stay_id", "ep_start",
                                       "note_offsets_h", "report_paths")]
            assert len(label_cols) == 25, f"expected 25 phenotype labels, got {len(label_cols)}"
            labels = cohort[label_cols].to_numpy(dtype=np.float32)

        samples = []
        for i in range(len(cohort)):
            row = cohort.iloc[i]
            tt, feats, mask = load_episode(split_dir, row["stay"], cfg["window"])
            if len(tt) == 0:
                continue
            notes, offs = load_notes(row)
            if not notes:
                continue
            samples.append({
                "name": row["stay"],
                "irg_ts": feats, "irg_ts_mask": mask, "ts_tt": tt,
                "text_data": notes,
                "text_time_to_end": cfg["window"] - offs,
                "label": labels[i],
            })
        splits[split] = samples
        n_notes = sum(len(s["text_data"]) for s in samples)
        print(f"[{task}/{split}] samples {len(samples)}, notes {n_notes}, "
              f"notes/sample {n_notes/len(samples):.2f}")

    # z-score stats from train masked observations, then normalize all splits
    acc = [[] for _ in CHANNELS]
    for s in splits["train"]:
        for j in range(len(CHANNELS)):
            v = s["irg_ts"][s["irg_ts_mask"][:, j] == 1, j]
            if len(v):
                acc[j].append(v)
    means, stds = [], []
    for j in range(len(CHANNELS)):
        v = np.concatenate(acc[j]) if acc[j] else np.zeros(1)
        means.append(float(v.mean()))
        stds.append(float(max(v.std(), 1e-6)))
    means = np.array(means, dtype=np.float32)
    stds = np.array(stds, dtype=np.float32)
    for split, samples in splits.items():
        for s in samples:
            obs = s["irg_ts_mask"] == 1
            s["irg_ts"] = np.where(obs, (s["irg_ts"] - means) / stds, 0.0).astype(np.float32)
        with open(os.path.join(out_dir, f"{split}_p2x.pkl"), "wb") as f:
            pickle.dump(samples, f, protocol=4)
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump({"channels": CHANNELS, "mean": means.tolist(), "std": stds.tolist()}, f, indent=1)
    print(f"[{task}] wrote pickles to {out_dir}")


if __name__ == "__main__":
    for t in (sys.argv[1:] or list(TASKS)):
        build(t)
