"""Step 3: precompute frozen BioClinicalBERT CLS embeddings for every note.

The FedCMA paper's backbone (MultimodalMIMIC) encodes each note with a
clinical BERT and takes the [CLS] vector (768-d). To keep federated training
tractable we freeze the text encoder: embeddings are precomputed once here
and the text branch trains the time-attention projection on top of them.

Outputs data/{task}/notes_{split}.npz:
  emb      float16 [n_notes_unique, 768]
  note_ids per-sample lists mapping into emb (pickle part in meta.pkl)
"""
import os
import pickle
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODEL = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN = 256
BATCH = 64


@torch.no_grad()
def embed_unique(texts, tok, model, device):
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    emb = np.zeros((len(texts), 768), dtype=np.float16)
    for b0 in range(0, len(order), BATCH):
        idx = order[b0:b0 + BATCH]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt").to(device)
        out = model(**enc).last_hidden_state[:, 0, :]   # [CLS]
        emb[idx] = out.cpu().to(torch.float16).numpy()
    return emb


def run(task):
    out_dir = os.path.join(DATA, task)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()

    for split in ["train", "val", "test"]:
        with open(os.path.join(out_dir, f"{split}_p2x.pkl"), "rb") as f:
            samples = pickle.load(f)
        texts, ids = [], {}
        for s in samples:
            s["note_ids"] = []
            for t in s["text_data"]:
                if t not in ids:
                    ids[t] = len(texts)
                    texts.append(t)
                s["note_ids"].append(ids[t])
        emb = embed_unique(texts, tok, model, device)
        np.savez_compressed(os.path.join(out_dir, f"notes_{split}.npz"), emb=emb)
        with open(os.path.join(out_dir, f"{split}_meta.pkl"), "wb") as f:
            pickle.dump([{"name": s["name"], "note_ids": s["note_ids"]} for s in samples], f)
        print(f"[{task}/{split}] unique notes {len(texts)} -> notes_{split}.npz "
              f"(emb {emb.shape})")


if __name__ == "__main__":
    for t in (sys.argv[1:] or ["ihm", "pheno"]):
        run(t)
