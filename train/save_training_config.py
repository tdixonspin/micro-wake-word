"""
Cell 9 – Write the training configuration YAML file.

Saves hyperparameters that control the training process to
``training_parameters.yaml``.  Experiment with sampling/penalty weights and
the number of training steps to improve model quality.
"""

import os
from pathlib import Path

import yaml

REAL_POSITIVE_SAMPLING_WEIGHT = 2.0
REAL_NEGATIVE_SAMPLING_WEIGHT = 5.0


def _contains_mmaps(features_dir: str) -> bool:
    training_dir = os.path.join(features_dir, "training")
    return any(Path(training_dir).glob("**/*_mmap"))


def run(target_word: str) -> None:  # noqa: ARG001
    """Write training_parameters.yaml with default hyperparameters."""

    config = {}

    config["window_step_ms"] = 10

    config["train_dir"] = "trained_models/wakeword"

    # Feature directories – each entry must contain training/, validation/,
    # testing/, or validation_ambient/ sub-folders with ragged-mmap folders.
    config["features"] = [
        {
            "features_dir": "generated_augmented_features",
            "sampling_weight": 2.0,
            "penalty_weight": 1.0,
            "truth": True,
            "truncation_strategy": "truncate_start",
            "type": "mmap",
        },
        {
            "features_dir": "negative_datasets/speech",
            "sampling_weight": 10.0,
            "penalty_weight": 1.0,
            "truth": False,
            "truncation_strategy": "random",
            "type": "mmap",
        },
        {
            "features_dir": "negative_datasets/dinner_party",
            "sampling_weight": 10.0,
            "penalty_weight": 1.0,
            "truth": False,
            "truncation_strategy": "random",
            "type": "mmap",
        },
        {
            "features_dir": "negative_datasets/no_speech",
            "sampling_weight": 5.0,
            "penalty_weight": 1.0,
            "truth": False,
            "truncation_strategy": "random",
            "type": "mmap",
        },
        {   # Only used for validation and testing
            "features_dir": "negative_datasets/dinner_party_eval",
            "sampling_weight": 0.0,
            "penalty_weight": 1.0,
            "truth": False,
            "truncation_strategy": "split",
            "type": "mmap",
        },
    ]

    if _contains_mmaps("generated_real_positive_features"):
        config["features"].append(
            {
                "features_dir": "generated_real_positive_features",
                "sampling_weight": REAL_POSITIVE_SAMPLING_WEIGHT,
                "penalty_weight": 1.0,
                "truth": True,
                "truncation_strategy": "truncate_start",
                "type": "mmap",
            }
        )

    if _contains_mmaps("generated_real_negative_features"):
        config["features"].append(
            {
                "features_dir": "generated_real_negative_features",
                "sampling_weight": REAL_NEGATIVE_SAMPLING_WEIGHT,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            }
        )

    # Training steps and corresponding per-step hyperparameters
    config["training_steps"] = [10000]
    config["positive_class_weight"] = [1]
    config["negative_class_weight"] = [20]
    config["learning_rates"] = [0.001]
    config["batch_size"] = 128

    # SpecAugment (one value per training-step entry)
    config["time_mask_max_size"] = [0]
    config["time_mask_count"] = [0]
    config["freq_mask_max_size"] = [0]
    config["freq_mask_count"] = [0]

    config["eval_step_interval"] = 500
    config["clip_duration_ms"] = 1500

    # Model-selection strategy
    config["target_minimization"] = 0.9
    config["minimization_metric"] = None   # Set to a metric name to enable
    config["maximization_metric"] = "average_viable_recall"

    with open("training_parameters.yaml", "w") as fh:
        yaml.dump(config, fh)

    print("Saved training_parameters.yaml")
