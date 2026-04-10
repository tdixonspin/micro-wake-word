"""
Cell 1 – Install microWakeWord and its dependencies.

Be sure to restart the Python session after this step if running interactively.
"""

import platform
import subprocess
import sys


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

    # piper-sample-generator is now a pip package (no git clone required)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "piper-sample-generator"],
        check=True,
    )

    # Install microWakeWord from the repository root (we are already inside it)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        check=True,
    )
