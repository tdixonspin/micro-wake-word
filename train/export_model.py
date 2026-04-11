"""
Cell 11 – Report the location of the finished TFLite model file.

After training the quantised streaming model is written to:
  trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite

To use this in ESPHome you must author a Model JSON manifest file.
See https://esphome.io/components/micro_wake_word for details and
https://github.com/esphome/micro-wake-word-models/tree/main/models/v2 for
examples.  Adjust the probability threshold based on the test results printed
at the end of training.
"""

import os


TFLITE_PATH = (
    "trained_models/wakeword"
    "/tflite_stream_state_internal_quant"
    "/stream_state_internal_quant.tflite"
)


def run(target_word: str) -> None:  # noqa: ARG001
    """Print the path to the final TFLite model and confirm it exists."""

    abs_path = os.path.abspath(TFLITE_PATH)
    if os.path.exists(abs_path):
        print(f"Training complete!  TFLite model: {abs_path}")
    else:
        print(
            f"WARNING: expected TFLite model not found at {abs_path}\n"
            "Training may not have completed successfully."
        )
