"""
begin_training.py – Entry point for the microWakeWord training pipeline.

Runs each training step in the order they appear in the original notebook:

  1.  install                   – Install microWakeWord and dependencies
  2.  generate_single_sample    – Generate one TTS sample for verification
  3.  generate_samples          – Generate 1000 TTS samples
  4.  download_augmentation_data– Download RIR / AudioSet / FMA data
  5.  setup_augmentation        – Validate augmentation configuration
  6.  verify_augmentation       – Augment one clip for a sanity check
  7.  generate_augmented_features – Build training / validation / test mmaps
  8.  download_negative_datasets– Download negative spectrogram datasets
  9.  save_training_config      – Write training_parameters.yaml
  10. train_model               – Train and convert the model
  11. export_model              – Report the final TFLite file location

Usage
-----
  python train/begin_training.py --target_word "khum_puter"

The script changes the working directory to the repository root so that all
relative paths used by the individual steps resolve correctly regardless of
where the script is invoked from.
"""

import argparse
import os
import sys

# Ensure the repository root (parent of this file's directory) is on sys.path
# so that ``from train.xxx import ...`` works inside the step modules.
_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TRAIN_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Change cwd to the repo root so all relative paths in step modules resolve.
os.chdir(_REPO_ROOT)

from train import (  # noqa: E402 – imports after sys.path manipulation
    download_augmentation_data,
    download_negative_datasets,
    export_model,
    generate_augmented_features,
    generate_samples,
    generate_single_sample,
    install,
    save_training_config,
    setup_augmentation,
    train_model,
    verify_augmentation,
)

STEPS = [
    ("1/11  Install dependencies",              install),
    ("2/11  Generate single verification sample", generate_single_sample),
    ("3/11  Generate training samples",          generate_samples),
    ("4/11  Download augmentation data",         download_augmentation_data),
    ("5/11  Set up augmentation",               setup_augmentation),
    ("6/11  Verify augmentation",               verify_augmentation),
    ("7/11  Generate augmented features",        generate_augmented_features),
    ("8/11  Download negative datasets",         download_negative_datasets),
    ("9/11  Save training config",              save_training_config),
    ("10/11 Train model",                       train_model),
    ("11/11 Export model",                      export_model),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full microWakeWord training pipeline for a given target word."
    )
    parser.add_argument(
        "--target_word",
        required=True,
        help=(
            "The word (or phonetic spelling) to train a wake-word model for. "
            "Phonetic spellings often produce better TTS samples, e.g. 'khum_puter'."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_word = args.target_word

    print(f"\n=== microWakeWord training pipeline ===")
    print(f"Target word: {target_word!r}")
    print(f"Working directory: {os.getcwd()}\n")

    for label, module in STEPS:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")
        module.run(target_word)

    print("\nAll steps completed successfully.")


if __name__ == "__main__":
    main()
