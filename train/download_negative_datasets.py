"""
Cell 8 – Download pre-generated negative spectrogram datasets from HuggingFace.

Downloads and extracts four datasets into ./negative_datasets/:
  - dinner_party
  - dinner_party_eval
  - no_speech
  - speech
"""

import os
import subprocess


def run(target_word: str) -> None:  # noqa: ARG001
    """Download negative spectrogram datasets (skipped if already present)."""

    output_dir = "./negative_datasets"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        link_root = "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/"
        filenames = [
            "dinner_party.zip",
            "dinner_party_eval.zip",
            "no_speech.zip",
            "speech.zip",
        ]
        for fname in filenames:
            link = link_root + fname
            zip_path = f"negative_datasets/{fname}"
            subprocess.run(["wget", "-O", zip_path, link], check=True)
            subprocess.run(["unzip", "-q", zip_path, "-d", output_dir], check=True)
