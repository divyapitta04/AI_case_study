"""Learned damage classifier: frozen DINOv2-small features + a linear (softmax) head.

The head is trained on real photos of damaged building surfaces (BD3) plus non-damage candidates from DEV rooms
(dataprep/train_damage_head.py) and stored in floorscan/models/damage_head.npz.  At run time `classify` gives class
probabilities for image crops; damage.py uses it to verify the classical candidates (furniture, pictures and shadows ->
"none") and to name the class.  Model weights: facebook/dinov2-small (Apache-2.0), fetched once by scripts/fetch_models.py.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

CLASSES: List[str] = ["water_stain", "mold", "hole", "peeling", "crack", "none"]
MODEL = "facebook/dinov2-small"
HEAD_PATH = os.path.join(os.path.dirname(__file__), "models", "damage_head.npz")


class Embedder:
    """DINOv2-small, CPU: image list -> (N, 768) features = [CLS token, mean patch token]."""

    def __init__(self, batch: int = 32):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(MODEL)
        self.model = AutoModel.from_pretrained(MODEL).eval()
        self.batch = batch

    def __call__(self, images) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(images), self.batch):
                x = self.proc(images=images[i:i + self.batch], return_tensors="pt")
                h = self.model(**x).last_hidden_state
                out.append(self.torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 768), np.float32)


def train_head(X: np.ndarray, y: np.ndarray, l2: float = 1e-2) -> Dict[str, np.ndarray]:
    """Multinomial logistic regression with L2 on standardised features (L-BFGS)."""
    from scipy.optimize import minimize
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    n, d = Z.shape
    k = len(CLASSES)
    Y = np.zeros((n, k))
    Y[np.arange(n), y] = 1
    cw = n / (k * np.maximum(Y.sum(0), 1))  # balance classes
    sw = (Y * cw).sum(1)

    def f(theta):
        W, b = theta[:d * k].reshape(d, k), theta[d * k:]
        S = Z @ W + b
        S -= S.max(1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(1, keepdims=True)
        loss = -(sw * np.log(np.maximum((P * Y).sum(1), 1e-12))).sum() / n + 0.5 * l2 * (W ** 2).sum()
        G = (P - Y) * sw[:, None] / n
        return loss, np.concatenate([(Z.T @ G + l2 * W).ravel(), G.sum(0)])

    r = minimize(f, np.zeros(d * k + k), jac=True, method="L-BFGS-B", options=dict(maxiter=500))
    return dict(W=r.x[:d * k].reshape(d, k).astype(np.float32), b=r.x[d * k:].astype(np.float32),
                mu=mu.astype(np.float32), sd=sd.astype(np.float32))


def predict_proba(head, X: np.ndarray) -> np.ndarray:
    S = ((X - head["mu"]) / head["sd"]) @ head["W"] + head["b"]
    S -= S.max(1, keepdims=True)
    P = np.exp(S)
    return P / P.sum(1, keepdims=True)


_CACHE: Dict[str, object] = {}


def available() -> bool:
    if not os.path.exists(HEAD_PATH):
        return False
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def classify(images) -> Optional[np.ndarray]:
    """Class probabilities (N, len(CLASSES)) for a list of RGB crops (PIL or uint8 arrays); None if the model is missing."""
    if not available() or not len(images):
        return None
    from PIL import Image
    if "emb" not in _CACHE:
        _CACHE["emb"] = Embedder()
        _CACHE["head"] = dict(np.load(HEAD_PATH))
    ims = [im if isinstance(im, Image.Image) else Image.fromarray(np.asarray(im, np.uint8)) for im in images]
    return predict_proba(_CACHE["head"], _CACHE["emb"](ims))
