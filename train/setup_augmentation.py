"""
Cell 5 – Configure the Clips and Augmentation objects.

Returns a (clips, augmenter) tuple that downstream steps can reuse, and also
exposes a top-level ``run()`` entry-point for the sequential training pipeline.
"""

from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips


def build_clips_and_augmenter() -> tuple:
    """Construct and return (Clips, Augmentation) with default settings."""

    clips = Clips(
        input_directory="generated_samples",
        file_pattern="*.wav",
        max_clip_duration_s=None,
        remove_silence=False,
        random_split_seed=10,
        split_count=0.1,
    )

    augmenter = Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.1,
            "TanhDistortion": 0.1,
            "PitchShift": 0.1,
            "BandStopFilter": 0.1,
            "AddColorNoise": 0.1,
            "AddBackgroundNoise": 0.75,
            "Gain": 1.0,
            "RIR": 0.5,
        },
        impulse_paths=["mit_rirs"],
        background_paths=["fma_16k", "audioset_16k"],
        background_min_snr_db=-5,
        background_max_snr_db=10,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )

    return clips, augmenter


def run(target_word: str) -> None:  # noqa: ARG001
    """Build augmentation objects (validation only – no file output)."""

    clips, augmenter = build_clips_and_augmenter()
    print(
        f"Augmentation setup complete: {len(clips)} clips loaded, augmenter ready."
    )
