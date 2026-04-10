"""
Cell 2 – Generate a single TTS sample of the target word for manual verification.

Installs piper-sample-generator (if needed), downloads the LibriTTS-R generator
model, and produces one wav file so you can confirm the phonetic spelling sounds
correct before generating thousands of samples.

The generator model is downloaded to ./piper-sample-generator/models/en-us-libritts-high.pt on first run.
"""

import os
import subprocess
import sys

from train.install import PIPER_SG_DIR

# Store the model inside the cloned repo's models/ directory so that the
# matching en-us-libritts-high.pt.json config (committed in that repo) is
# automatically present alongside it.
MODEL_DIR = os.path.join(PIPER_SG_DIR, "models")
MODEL_FILENAME = "en-us-libritts-high.pt"
MODEL_URL = (
    "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0"
    "/en_US-libritts_r-medium.pt"
)


def _ensure_model() -> str:
    """Download the LibriTTS-R generator model if not already present.

    Returns the local path to the model file.
    """
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    if not os.path.exists(model_path):
        subprocess.run(["wget", "-O", model_path, MODEL_URL], check=True)
    return model_path


def run(target_word: str) -> None:
    """Generate one verification sample for *target_word*."""

    model_path = _ensure_model()

    # piper_train/ lives at the root of the piper-sample-generator clone,
    # one level above the piper_sample_generator/ package directory.
    # Set PYTHONPATH so the subprocess can find it.
    env = os.environ.copy()
    env["PYTHONPATH"] = PIPER_SG_DIR + os.pathsep + env.get("PYTHONPATH", "")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "piper_sample_generator",
            target_word,
            "--model",
            os.path.abspath(model_path),
            "--max-samples",
            "1",
            "--batch-size",
            "1",
            "--output-dir",
            "generated_samples",
        ],
        env=env,
        check=True,
    )

    print("Verification sample saved to generated_samples/0.wav")
