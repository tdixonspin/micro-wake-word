"""
Cell 2 – Generate a single TTS sample of the target word for manual verification.

Clones piper-sample-generator (if absent) and produces one wav file so you can
confirm the phonetic spelling sounds correct before generating thousands of samples.
"""

import os
import platform
import subprocess
import sys


def run(target_word: str) -> None:
    """Generate one verification sample for *target_word*."""

    if not os.path.exists("./piper-sample-generator"):
        if platform.system() == "Darwin":
            subprocess.run(
                [
                    "git",
                    "clone",
                    "-b",
                    "mps-support",
                    "https://github.com/kahrendt/piper-sample-generator",
                ],
                check=True,
            )
        else:
            subprocess.run(
                ["git", "clone", "https://github.com/rhasspy/piper-sample-generator"],
                check=True,
            )

        subprocess.run(
            [
                "wget",
                "-O",
                "piper-sample-generator/models/en_US-libritts_r-medium.pt",
                "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt",
            ],
            check=True,
        )

        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch",
                "torchaudio",
                "piper-phonemize-cross==1.2.1",
            ],
            check=True,
        )

    if "piper-sample-generator/" not in sys.path:
        sys.path.append("piper-sample-generator/")

    subprocess.run(
        [
            sys.executable,
            "piper-sample-generator/generate_samples.py",
            target_word,
            "--max-samples",
            "1",
            "--batch-size",
            "1",
            "--output-dir",
            "generated_samples",
        ],
        check=True,
    )

    print(f"Verification sample saved to generated_samples/0.wav")
