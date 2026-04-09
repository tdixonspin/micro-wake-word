"""
Cell 7 – Generate augmented spectrogram features for training, validation, and testing.

Writes mmap datasets under ./generated_augmented_features/:
  training/wakeword_mmap
  validation/wakeword_mmap
  testing/wakeword_mmap
"""

import os

from mmap_ninja.ragged import RaggedMmap

from microwakeword.audio.spectrograms import SpectrogramGeneration
from train.setup_augmentation import build_clips_and_augmenter


def run(target_word: str) -> None:  # noqa: ARG001
    """Produce augmented spectrograms for all three data splits."""

    clips, augmenter = build_clips_and_augmenter()

    output_dir = "generated_augmented_features"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    split_configs = [
        # (split_name, mmap_split_name, repetition, slide_frames)
        ("training", "train", 2, 10),
        ("validation", "validation", 1, 10),
        ("testing", "test", 1, 1),
    ]

    for split, split_name, repetition, slide_frames in split_configs:
        out_dir = os.path.join(output_dir, split)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        spectrograms = SpectrogramGeneration(
            clips=clips,
            augmenter=augmenter,
            slide_frames=slide_frames,
            step_ms=10,
        )

        RaggedMmap.from_generator(
            out_dir=os.path.join(out_dir, "wakeword_mmap"),
            sample_generator=spectrograms.spectrogram_generator(
                split=split_name, repeat=repetition
            ),
            batch_size=100,
            verbose=True,
        )
