"""
Generate spectrogram mmap datasets from real positive and negative audio samples.

Input directories are passed via:
  - MWW_REAL_POSITIVE_DIRS
  - MWW_REAL_NEGATIVE_DIRS

Each variable should contain one or more directories separated by ``os.pathsep``.
"""

import os
import shutil
from pathlib import Path

from mmap_ninja.ragged import RaggedMmap

from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration

REAL_POSITIVE_DIRS_ENV = "MWW_REAL_POSITIVE_DIRS"
REAL_NEGATIVE_DIRS_ENV = "MWW_REAL_NEGATIVE_DIRS"
MIN_WAVS_FOR_DATA_SPLIT = 5
DATA_SPLIT_SEED = 10
DATA_SPLIT_RATIO = 0.1


def _read_input_dirs(env_var: str) -> list[str]:
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return []

    paths = [path for path in raw.split(os.pathsep) if path]
    for input_dir in paths:
        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"Directory does not exist: {input_dir}")
    return paths


def _generate_dataset(
    input_dirs: list[str],
    output_root: str,
    prefix: str,
) -> bool:
    if not input_dirs:
        return False

    split_configs = [
        ("training", "train"),
        ("validation", "validation"),
        ("testing", "test"),
    ]

    generated = False
    for index, input_dir in enumerate(input_dirs, start=1):
        wav_paths = list(Path(input_dir).glob("**/*.wav"))
        if not wav_paths:
            print(f"Skipping directory with no wav files: {input_dir}")
            continue

        use_split = len(wav_paths) >= MIN_WAVS_FOR_DATA_SPLIT
        clips = Clips(
            input_directory=input_dir,
            file_pattern="**/*.wav",
            max_clip_duration_s=None,
            remove_silence=False,
            random_split_seed=DATA_SPLIT_SEED if use_split else None,
            split_count=DATA_SPLIT_RATIO,
        )

        spectrograms = SpectrogramGeneration(
            clips=clips,
            augmenter=None,
            slide_frames=1,
            step_ms=10,
        )

        for split, split_name in split_configs:
            split_dir = os.path.join(output_root, split)
            os.makedirs(split_dir, exist_ok=True)
            mmap_dir = os.path.join(split_dir, f"{prefix}_{index:02d}_mmap")
            if os.path.exists(mmap_dir):
                shutil.rmtree(mmap_dir)

            if use_split:
                sample_generator = spectrograms.spectrogram_generator(
                    split=split_name,
                    repeat=1,
                )
            else:
                sample_generator = spectrograms.spectrogram_generator(repeat=1)

            RaggedMmap.from_generator(
                out_dir=mmap_dir,
                sample_generator=sample_generator,
                batch_size=100,
                verbose=True,
            )

        generated = True
        print(f"Generated real-sample mmap features from {input_dir}")

    return generated


def run(target_word: str) -> None:  # noqa: ARG001
    """Generate mmap features from real positive and negative samples."""

    positive_dirs = _read_input_dirs(REAL_POSITIVE_DIRS_ENV)
    negative_dirs = _read_input_dirs(REAL_NEGATIVE_DIRS_ENV)

    if not positive_dirs and not negative_dirs:
        print("No real sample directories provided; skipping real feature generation.")
        return

    positive_generated = _generate_dataset(
        input_dirs=positive_dirs,
        output_root="generated_real_positive_features",
        prefix="real_positive",
    )
    negative_generated = _generate_dataset(
        input_dirs=negative_dirs,
        output_root="generated_real_negative_features",
        prefix="real_negative",
    )

    if not positive_generated and not negative_generated:
        print("No real wav files found in the provided directories.")
