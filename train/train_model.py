"""
Cell 10 – Train the model and convert it to a streaming TFLite file.

Invokes ``microwakeword.model_train_eval`` with a MixedNet architecture.
The run will resume from the last checkpoint if interrupted.  Pass
``--train 0`` on the command line to skip training and only convert/test
the best-weighted model.
"""

import subprocess
import sys


def run(target_word: str) -> None:  # noqa: ARG001
    """Run model training and quantised streaming conversion."""

    subprocess.run(
        [
            sys.executable,
            "-m",
            "microwakeword.model_train_eval",
            "--training_config=training_parameters.yaml",
            "--train",
            "1",
            "--restore_checkpoint",
            "1",
            "--test_tf_nonstreaming",
            "0",
            "--test_tflite_nonstreaming",
            "0",
            "--test_tflite_nonstreaming_quantized",
            "0",
            "--test_tflite_streaming",
            "0",
            "--test_tflite_streaming_quantized",
            "1",
            "--use_weights",
            "best_weights",
            "mixednet",
            "--pointwise_filters",
            "64,64,64,64",
            "--repeat_in_block",
            "1, 1, 1, 1",
            "--mixconv_kernel_sizes",
            "[5], [7,11], [9,15], [23]",
            "--residual_connection",
            "0,0,0,0",
            "--first_conv_filters",
            "32",
            "--first_conv_kernel_size",
            "5",
            "--stride",
            "3",
        ],
        check=True,
    )
