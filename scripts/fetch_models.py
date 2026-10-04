"""Fetch the pretrained weights: monocular depth (photo / video tiers) and DINOv2-small (damage classifier). Run once.

    python scripts/fetch_models.py

Model: depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf (Apache-2.0, ~100 MB), loaded via HuggingFace
transformers (installed by requirements.txt; installed here too if missing).
No call is made to our infrastructure: weights come from the public HuggingFace hub once, then run offline.
"""
import subprocess
import sys

MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DAMAGE_MODEL = "facebook/dinov2-small"  # damage classifier features (Apache-2.0)


def main():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        print("installing torch (CPU) + transformers ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "transformers",
                               "--index-url", "https://download.pytorch.org/whl/cpu",
                               "--extra-index-url", "https://pypi.org/simple"])
    from transformers import AutoImageProcessor, AutoModel, pipeline
    pipeline(task="depth-estimation", model=MODEL, device=-1)
    print("model cached:", MODEL)
    AutoImageProcessor.from_pretrained(DAMAGE_MODEL)
    AutoModel.from_pretrained(DAMAGE_MODEL)
    print("model cached:", DAMAGE_MODEL)


if __name__ == "__main__":
    main()
