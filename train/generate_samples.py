"""
Cell 3 – Generate a larger set of TTS wake-word samples.

Runs piper-sample-generator to produce 1000 wav files.  Start here when
iterating to improve model quality – experiment with noise-scales,
noise-scale-ws, and generating negative samples similar to the wake word.
"""

import subprocess
import sys


def run(target_word: str) -> None:
    """Generate 1000 TTS samples of *target_word* into ./generated_samples."""

    subprocess.run(
        [
            sys.executable,
            "piper-sample-generator/generate_samples.py",
            target_word,
            "--max-samples",
            "1000",
            "--batch-size",
            "100",
            "--output-dir",
            "generated_samples",
        ],
        check=True,
    )
