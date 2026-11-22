"""
TFLite-only prediction module for BirdNET.

This module provides a lightweight prediction interface that uses only
tflite-runtime, avoiding the heavy dependencies on keras and full TensorFlow.
"""

import numpy as np

try:
    import tflite_runtime.interpreter as tflite  # type: ignore
except ModuleNotFoundError:
    from tensorflow import lite as tflite

# BirdNET model expects exactly this many samples (3 seconds at 48kHz)
DEFAULT_SAMPLE_COUNT = 144000


class TFLitePredictor:
    """Lightweight TFLite predictor for BirdNET models.

    The interpreter is sized for a single fixed-length sample (batch size 1)
    and its tensors are allocated exactly once at load time. Re-allocating
    tensors per call is one of the most expensive TFLite operations, so
    ``predict`` only copies the input in and invokes the model.
    """

    def __init__(
        self,
        model_path: str,
        num_threads: int = 1,
        sample_count: int = DEFAULT_SAMPLE_COUNT,
    ):
        """
        Initialize the TFLite predictor.

        Args:
            model_path: Path to the .tflite model file
            num_threads: Number of threads to use for inference
            sample_count: Number of audio samples per prediction
        """
        self.model_path = model_path
        self.num_threads = num_threads
        self.sample_count = sample_count
        self.interpreter = None
        self.input_index = None
        self.output_index = None

        self._load_model()

    def _load_model(self):
        """Load the TFLite model and allocate tensors once for shape (1, sample_count)."""
        self.interpreter = tflite.Interpreter(
            model_path=self.model_path,
            num_threads=self.num_threads,
        )

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.input_index = input_details[0]["index"]
        self.output_index = output_details[0]["index"]

        wanted_shape = [1, self.sample_count]
        if list(input_details[0]["shape"]) != wanted_shape:
            self.interpreter.resize_tensor_input(self.input_index, wanted_shape)
        self.interpreter.allocate_tensors()

    def predict(self, sample: np.ndarray) -> np.ndarray:
        """
        Predict bird species scores for one audio sample.

        Args:
            sample: float32 array of shape (sample_count,) containing audio
                    normalised to [-1, 1]

        Returns:
            Raw model scores, shape (num_classes,)
        """
        # reshape is a view, set_tensor copies straight into the interpreter
        self.interpreter.set_tensor(self.input_index, sample.reshape(1, -1))
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)[0]


# Global predictor instance
_predictor = None


def init_predictor(
    model_path: str, num_threads: int = 1, sample_count: int = DEFAULT_SAMPLE_COUNT
) -> TFLitePredictor:
    """
    Initialize the global predictor instance.

    Args:
        model_path: Path to the .tflite model file
        num_threads: Number of threads to use for inference
        sample_count: Number of audio samples per prediction

    Returns:
        The initialized TFLitePredictor instance
    """
    global _predictor
    _predictor = TFLitePredictor(model_path, num_threads, sample_count)
    return _predictor


def predict(sample: np.ndarray) -> np.ndarray:
    """
    Predict bird species scores for one audio sample using the global predictor.

    Args:
        sample: float32 array of shape (sample_count,) normalised to [-1, 1]

    Returns:
        Raw model scores, shape (num_classes,)

    Raises:
        RuntimeError: If the predictor has not been initialized
    """
    if _predictor is None:
        raise RuntimeError(
            "Predictor not initialized. Call init_predictor() first with the model path."
        )

    return _predictor.predict(sample)


def flat_sigmoid(x: np.ndarray, sensitivity: float = -1, bias: float = 1.0) -> np.ndarray:
    """
    Apply a flat sigmoid function to convert logits to probabilities.

    The flat sigmoid function is defined as:
        f(x) = 1 / (1 + exp(sensitivity * clip(x + bias, -20, 20)))

    We transform the bias parameter to a range of [-100, 100] with the formula:
        transformed_bias = (bias - 1.0) * 10.0

    Args:
        x: Input data (logits from the model)
        sensitivity: Sensitivity parameter for the sigmoid function. Default is -1.
        bias: Bias parameter to shift the sigmoid function. Default is 1.0.

    Returns:
        Transformed data after applying the flat sigmoid function
    """
    transformed_bias = (bias - 1.0) * 10.0
    return 1 / (1.0 + np.exp(sensitivity * np.clip(x + transformed_bias, -20, 20)))
