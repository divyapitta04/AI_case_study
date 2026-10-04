"""Train and evaluate the damage classifier head on REAL photos of damaged wall surfaces (BD3 dataset).

    python dataprep/train_damage_head.py [--bd3 ../arkitscenes_raw/bd3]

Data: BD3 - Building Defect Dataset (Kottari & Arjunan, 2024; CC-BY-4.0 via the HuggingFace mirror
chandrabhuma/building_defect_vqa): 512x512 photos of real building surfaces in 7 classes.  Its own train split (3172 images)
trains the head; its test split (793 images) is used only for the reported accuracy.
Features: DINOv2-small (facebook/dinov2-small, Apache-2.0), frozen, CLS token + mean patch token (768 values).
Head: multinomial logistic regression (L2), our classes
    water_stain <- stain,  mold <- algae,  hole <- spalling,  peeling <- peeling,  crack <- minor_crack + major_crack,
    none <- plain  (+ optional "none" negatives: candidate regions from undamaged DEV rooms - furniture, pictures, shadows).
Writes floorscan/models/damage_head.npz and results/damage/BD3_EVAL.md.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HF = "https://huggingface.co/datasets/chandrabhuma/building_defect_vqa/resolve/main/data/"
FILES = {"train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"], "test": ["test-00000-of-00001.parquet"]}
MAP = {"stain": "water_stain", "algae": "mold", "spalling": "hole", "peeling": "peeling", "minor_crack": "crack",
       "major_crack": "crack", "plain": "none"}


def load_split(bd3: str, split: str):
    import pyarrow.parquet as pq
    from PIL import Image
    imgs, labels = [], []
    for f in FILES[split]:
        p = os.path.join(bd3, f)
        if not os.path.exists(p):
            os.makedirs(bd3, exist_ok=True)
            urllib.request.urlretrieve(HF + f, p)
        for r in pq.read_table(p).to_pylist():
            imgs.append(Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"))
            labels.append(MAP[r["answer"]])
    return imgs, labels


def main():
    from floorscan.damage_model import CLASSES, Embedder, train_head
    ap = argparse.ArgumentParser()
    ap.add_argument("--bd3", default=os.path.join(ROOT, "..", "arkitscenes_raw", "bd3"))
    ap.add_argument("--negatives", help="npz with 'emb' of candidate regions from undamaged DEV rooms (class none)")
    a = ap.parse_args()
    emb = Embedder()
    feats = {}
    for split in ("train", "test"):
        imgs, labels = load_split(a.bd3, split)
        X = emb(imgs)
        feats[split] = (X, np.array([CLASSES.index(l) for l in labels]))
        print(split, X.shape, flush=True)
    Xtr, ytr = feats["train"]
    n_neg = 0
    if a.negatives and os.path.exists(a.negatives):
        neg = np.load(a.negatives)["emb"]
        n_neg = len(neg)
        Xtr = np.concatenate([Xtr, neg])
        ytr = np.concatenate([ytr, np.full(len(neg), CLASSES.index("none"))])
    head = train_head(Xtr, ytr)
    os.makedirs(os.path.join(ROOT, "floorscan", "models"), exist_ok=True)
    np.savez(os.path.join(ROOT, "floorscan", "models", "damage_head.npz"), **head, classes=np.array(CLASSES),
             trained_on=f"BD3 train ({len(feats['train'][1])} real photos) + {n_neg} DEV-room non-damage candidates")
    # ---- evaluation on the untouched BD3 test split -----------------------------------------------
    from floorscan.damage_model import predict_proba
    Xte, yte = feats["test"]
    P = predict_proba(head, Xte)
    yp = P.argmax(1)
    acc = float(np.mean(yp == yte))
    k = len(CLASSES)
    cm = np.zeros((k, k), int)
    for t, p in zip(yte, yp):
        cm[t, p] += 1
    dmg_true, dmg_pred = yte != CLASSES.index("none"), yp != CLASSES.index("none")
    det_p = float((dmg_true & dmg_pred).sum() / max(dmg_pred.sum(), 1))
    det_r = float((dmg_true & dmg_pred).sum() / max(dmg_true.sum(), 1))
    L = ["# Damage classifier on real photos (BD3 test split, never used for training)\n",
         f"Images: {len(yte)} real photos of building surfaces (BD3, CC-BY-4.0). Features: DINOv2-small (frozen); head: "
         f"logistic regression trained on BD3 train ({len(feats['train'][1])} photos) + {n_neg} non-damage candidates from DEV rooms.\n",
         f"* **class accuracy: {100 * acc:.1f} %** ({k} classes: {', '.join(CLASSES)})",
         f"* damage vs no damage: precision {100 * det_p:.1f} %, recall {100 * det_r:.1f} %\n",
         "| true \\ predicted | " + " | ".join(CLASSES) + " | recall |", "|---" * (k + 2) + "|"]
    for i, c in enumerate(CLASSES):
        L.append(f"| {c} | " + " | ".join(str(x) for x in cm[i]) + f" | {100 * cm[i, i] / max(cm[i].sum(), 1):.0f} % |")
    L.append("| precision | " + " | ".join(f"{100 * cm[j, j] / max(cm[:, j].sum(), 1):.0f} %" for j in range(k)) + " | |")
    L.append("\nThis measures recognition of the damage *class* on real close-up photos. Metric extent on a real wall needs staged "
             "damage with measured ground truth, which no public dataset provides (docs/REAL_CAPTURE_TODO.md).")
    os.makedirs(os.path.join(ROOT, "results", "damage"), exist_ok=True)
    open(os.path.join(ROOT, "results", "damage", "BD3_EVAL.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(dict(accuracy=acc, detection_precision=det_p, detection_recall=det_r, n_test=int(len(yte)), classes=CLASSES,
                   confusion=cm.tolist()), open(os.path.join(ROOT, "results", "damage", "bd3_metrics.json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
