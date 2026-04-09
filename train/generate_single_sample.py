"""
Cell 2 – Generate a single TTS sample of the target word for manual verification.

Installs piper-sample-generator (if needed), downloads the LibriTTS-R generator
model, and produces one wav file so you can confirm the phonetic spelling sounds
correct before generating thousands of samples.

The generator model is downloaded to ./models/en-us-libritts-high.pt on first run.
"""

import os
import subprocess
import sys

MODEL_DIR = "models"
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

    subprocess.run(
        [
            sys.executable,
            "-m",
            "piper_sample_generator",
            target_word,
            "--model",
            model_path,
            "--max-samples",
            "1",
            "--batch-size",
            "1",
            "--output-dir",
            "generated_samples",
        ],
        check=True,
    )

    print("Verification sample saved to generated_samples/0.wav")
