"""
Cell 3 – Generate a larger set of TTS wake-word samples.

Runs piper-sample-generator to produce 1000 wav files.  Start here when
iterating to improve model quality – experiment with --length-scales (speaking
speeds), --slerp-weights (speaker blending), and generating negative samples
similar to the wake word.
"""

import subprocess
import sys

from train.generate_single_sample import MODEL_DIR, MODEL_FILENAME, _ensure_model


def run(target_word: str) -> None:
    """Generate 1000 TTS samples of *target_word* into ./generated_samples."""

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
            "1000",
            "--batch-size",
            "100",
            "--output-dir",
            "generated_samples",
        ],
        check=True,
    )
