"""
Cell 4 – Download audio data used for augmentation.

Downloads:
  - MIT environmental impulse responses  (./mit_rirs)
  - AudioSet balanced-train subset       (./audioset_16k)
  - Free Music Archive extra-small set   (./fma_16k)

**Important:** the data has a mixture of licences and usage restrictions.
Any model trained with this data should be considered appropriate for
**non-commercial personal use only**.
"""

import os
import subprocess

import datasets
import numpy as np
import scipy.io.wavfile
from pathlib import Path
from tqdm import tqdm


def run(target_word: str) -> None:  # noqa: ARG001
    """Download and resample all augmentation datasets."""

    # ------------------------------------------------------------------ #
    # MIT Room Impulse Responses
    # ------------------------------------------------------------------ #
    output_dir = "./mit_rirs"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        rir_dataset = datasets.load_dataset(
            "davidscripka/MIT_environmental_impulse_responses",
            split="train",
            streaming=True,
        )
        for idx, row in enumerate(tqdm(rir_dataset, desc="MIT RIRs")):
            name = f"rir_{idx:04d}.wav"
            scipy.io.wavfile.write(
                os.path.join(output_dir, name),
                16000,
                (row["audio"]["array"] * 32767).astype(np.int16),
            )

    # ------------------------------------------------------------------ #
    # AudioSet balanced-train subset
    # ------------------------------------------------------------------ #
    if not os.path.exists("audioset"):
        os.mkdir("audioset")

        fname = "bal_train09.tar"
        out_path = f"audioset/{fname}"
        link = "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/" + fname
        subprocess.run(["wget", "-O", out_path, link], check=True)
        subprocess.run(["tar", "-xf", fname], cwd="audioset", check=True)

        output_dir = "./audioset_16k"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        audioset_files = sorted(Path("audioset/audio").glob("**/*.flac"))
        audioset_dataset = datasets.Dataset.from_dict(
            {"audio": [str(f) for f in audioset_files]}
        )
        audioset_dataset = audioset_dataset.cast_column(
            "audio", datasets.Audio(sampling_rate=16000)
        )
        for file_path, row in tqdm(
            zip(audioset_files, audioset_dataset), desc="AudioSet", total=len(audioset_files)
        ):
            name = file_path.name.replace(".flac", ".wav")
            scipy.io.wavfile.write(
                os.path.join(output_dir, name),
                16000,
                (row["audio"]["array"] * 32767).astype(np.int16),
            )

    # ------------------------------------------------------------------ #
    # Free Music Archive extra-small set
    # ------------------------------------------------------------------ #
    output_dir = "./fma"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        fname = "fma_xs.zip"
        link = "https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/" + fname
        out_path = f"fma/{fname}"
        subprocess.run(["wget", "-O", out_path, link], check=True)
        subprocess.run(["unzip", "-q", fname], cwd=output_dir, check=True)

        fma_16k_dir = "./fma_16k"
        if not os.path.exists(fma_16k_dir):
            os.mkdir(fma_16k_dir)

        fma_files = sorted(Path("fma/fma_small").glob("**/*.mp3"))
        fma_dataset = datasets.Dataset.from_dict(
            {"audio": [str(f) for f in fma_files]}
        )
        fma_dataset = fma_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
        for file_path, row in tqdm(
            zip(fma_files, fma_dataset), desc="FMA", total=len(fma_files)
        ):
            name = file_path.name.replace(".mp3", ".wav")
            scipy.io.wavfile.write(
                os.path.join(fma_16k_dir, name),
                16000,
                (row["audio"]["array"] * 32767).astype(np.int16),
            )
