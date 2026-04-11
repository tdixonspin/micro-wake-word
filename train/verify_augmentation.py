"""
Cell 6 – Augment a random clip and save it for verification.

Saves the result to ``augmented_clip.wav`` in the working directory so you
can listen to it and confirm the augmentation pipeline sounds reasonable.
"""

from microwakeword.audio.audio_utils import save_clip
from train.setup_augmentation import build_clips_and_augmenter


def run(target_word: str) -> None:  # noqa: ARG001
    """Augment one random clip and write it to ./augmented_clip.wav."""

    clips, augmenter = build_clips_and_augmenter()
    random_clip = clips.get_random_clip()
    augmented_clip = augmenter.augment_clip(random_clip)
    save_clip(augmented_clip, "augmented_clip.wav")
    print("Verification clip saved to augmented_clip.wav")
