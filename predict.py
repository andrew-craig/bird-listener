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


class TFLitePredictor:
    """Lightweight TFLite predictor for BirdNET models."""

    def __init__(self, model_path: str, num_threads: int = 1):
        """
        Initialize the TFLite predictor.

        Args:
            model_path: Path to the .tflite model file
            num_threads: Number of threads to use for inference
        """
        self.model_path = model_path
        self.num_threads = num_threads
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_index = None
        self.output_index = None

        self._load_model()

    def _load_model(self):
        """Load the TFLite model and allocate tensors."""
        # Load TFLite model and allocate tensors
        self.interpreter = tflite.Interpreter(
            model_path=self.model_path,
            num_threads=self.num_threads,
        )
        self.interpreter.allocate_tensors()

        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Get input and output tensor indices
        self.input_index = self.input_details[0]["index"]
        self.output_index = self.output_details[0]["index"]

    def predict(self, samples: list[np.ndarray]) -> np.ndarray:
        """
        Predict bird species from audio samples.

        Args:
            samples: List of audio samples, each should be a 1D numpy array
                    of shape (144000,) containing audio data normalized to [-1, 1]

        Returns:
            Prediction scores as a numpy array of shape (batch_size, num_classes)
        """
        # Convert samples to numpy array
        data = np.array(samples, dtype="float32")

        # Reshape input tensor to match batch size
        batch_size = len(samples)
        input_shape = [batch_size, *samples[0].shape]
        self.interpreter.resize_tensor_input(self.input_index, input_shape)
        self.interpreter.allocate_tensors()

        # Set input tensor
        self.interpreter.set_tensor(self.input_index, data)

        # Run inference
        self.interpreter.invoke()

        # Get output tensor
        prediction = self.interpreter.get_tensor(self.output_index)

        return prediction


# Global predictor instance
_predictor = None


def init_predictor(model_path: str, num_threads: int = 1) -> TFLitePredictor:
    """
    Initialize the global predictor instance.

    Args:
        model_path: Path to the .tflite model file
        num_threads: Number of threads to use for inference

    Returns:
        The initialized TFLitePredictor instance
    """
    global _predictor
    _predictor = TFLitePredictor(model_path, num_threads)
    return _predictor


def predict(samples: list[np.ndarray]) -> np.ndarray:
    """
    Predict bird species from audio samples using the global predictor.

    Args:
        samples: List of audio samples, each should be a 1D numpy array
                of shape (144000,) containing audio data normalized to [-1, 1]

    Returns:
        Prediction scores as a numpy array of shape (batch_size, num_classes)

    Raises:
        RuntimeError: If the predictor has not been initialized
    """
    if _predictor is None:
        raise RuntimeError(
            "Predictor not initialized. Call init_predictor() first with the model path."
        )

    return _predictor.predict(samples)


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
