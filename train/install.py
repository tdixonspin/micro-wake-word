"""
Cell 1 – Install microWakeWord and its dependencies.

Be sure to restart the Python session after this step if running interactively.
"""

import os
import platform
import subprocess
import sys

PIPER_SG_DIR = "piper-sample-generator"


def run(target_word: str) -> None:  # noqa: ARG001 – target_word unused but kept for uniform signature
    """Install microWakeWord and required packages."""

    if platform.system() == "Darwin":
        # pymicro-features fork with macOS/minimum-C++ support
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "git+https://github.com/puddly/pymicro-features@puddly/minimum-cpp-version",
            ],
            check=True,
        )

    # audio-metadata fork that unpins attrs so it doesn't break Jupyter
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "git+https://github.com/whatsnowplaying/audio-metadata@d4ebb238e6a401bb1a5aaaac60c9e2b3cb30929f",
        ],
        check=True,
    )

    # Clone piper-sample-generator so that piper_train/ (a sibling of
    # piper_sample_generator/ in the repo) is available at runtime.
    # The PyPI wheel omits piper_train, so we must use the source tree.
    if not os.path.exists(PIPER_SG_DIR):
        if platform.system() == "Darwin":
            # macOS fork adds MPS (Apple Silicon) support
            subprocess.run(
                ["git", "clone", "-b", "mps-support",
                 "https://github.com/kahrendt/piper-sample-generator", PIPER_SG_DIR],
                check=True,
            )
        else:
            subprocess.run(
                ["git", "clone",
                 "https://github.com/rhasspy/piper-sample-generator", PIPER_SG_DIR],
                check=True,
            )

    # Install declared dependencies (torch, torchaudio, piper-tts, etc.) while
    # leaving the source tree in place so piper_train remains importable.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", PIPER_SG_DIR],
        check=True,
    )

    # Install microWakeWord from the repository root (we are already inside it)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        check=True,
    )
