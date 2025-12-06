"""
Species list prediction module for BirdNET.

This module provides geographic and temporal filtering of bird species
using the BirdNET metadata model. It predicts which species are likely
to be present at a given location and time of year.
"""

import numpy as np

try:
    import tflite_runtime.interpreter as tflite  # type: ignore
except ModuleNotFoundError:
    from tensorflow import lite as tflite


class SpeciesPredictor:
    """Predicts species lists based on geographic location and time of year."""

    def __init__(self, model_path: str, num_threads: int = 1):
        """
        Initialize the species predictor.

        Args:
            model_path: Path to the metadata .tflite model file
            num_threads: Number of threads to use for inference
        """
        self.model_path = model_path
        self.num_threads = num_threads
        self.interpreter = None
        self.input_index = None
        self.output_index = None

        self._load_model()

    def _load_model(self):
        """Load the TFLite metadata model and allocate tensors."""
        # Load TFLite model and allocate tensors
        self.interpreter = tflite.Interpreter(
            model_path=self.model_path,
            num_threads=self.num_threads,
        )
        self.interpreter.allocate_tensors()

        # Get input and output tensors
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Get input and output tensor indices
        self.input_index = input_details[0]["index"]
        self.output_index = output_details[0]["index"]

    def predict_filter(self, lat: float, lon: float, week: int) -> np.ndarray:
        """
        Predict the probability for each species at a given location and time.

        Args:
            lat: The latitude
            lon: The longitude
            week: The week of the year [1-48]. Use -1 for year-round.

        Returns:
            A numpy array of probabilities for all species
        """
        # Prepare metadata as sample
        sample = np.expand_dims(np.array([lat, lon, week], dtype="float32"), 0)

        # Run inference
        self.interpreter.set_tensor(self.input_index, sample)
        self.interpreter.invoke()

        return self.interpreter.get_tensor(self.output_index)[0]

    def explore(
        self, lat: float, lon: float, week: int, labels: list[str], threshold: float
    ) -> list[tuple[float, str]]:
        """
        Predict the species list for a location, sorted by probability.

        Args:
            lat: The latitude
            lon: The longitude
            week: The week of the year [1-48]. Use -1 for year-round.
            labels: List of all species labels
            threshold: Minimum probability threshold for filtering

        Returns:
            A sorted list of tuples with (probability, species_name)
        """
        # Get filter predictions
        l_filter = self.predict_filter(lat, lon, week)

        # Apply threshold
        l_filter = np.where(l_filter >= threshold, l_filter, 0)

        # Zip with labels
        l_filter = list(zip(l_filter, labels, strict=True))

        # Sort by filter value (descending)
        return sorted(l_filter, key=lambda x: x[0], reverse=True)

    def get_species_list(
        self,
        lat: float,
        lon: float,
        week: int,
        labels: list[str],
        threshold: float = 0.05,
        sort: bool = False,
    ) -> list[str]:
        """
        Get a filtered species list for a location and time.

        Args:
            lat: The latitude
            lon: The longitude
            week: The week of the year [1-48]. Use -1 for year-round.
            labels: List of all species labels
            threshold: Only values above or equal to threshold will be included
            sort: If the species list should be sorted alphabetically

        Returns:
            A list of species names that meet the threshold
        """
        # Get predictions with scores
        predictions = self.explore(lat, lon, week, labels, threshold)

        # Extract species names where probability >= threshold
        species_list = [species for prob, species in predictions if prob >= threshold]

        return sorted(species_list) if sort else species_list


# Global predictor instance
_predictor = None


def init_species_predictor(model_path: str, num_threads: int = 1) -> SpeciesPredictor:
    """
    Initialize the global species predictor instance.

    Args:
        model_path: Path to the metadata .tflite model file
        num_threads: Number of threads to use for inference

    Returns:
        The initialized SpeciesPredictor instance
    """
    global _predictor
    _predictor = SpeciesPredictor(model_path, num_threads)
    return _predictor


def get_species_list(
    lat: float,
    lon: float,
    week: int,
    labels: list[str],
    threshold: float = 0.05,
    sort: bool = False,
) -> list[str]:
    """
    Get a filtered species list using the global predictor.

    Args:
        lat: The latitude
        lon: The longitude
        week: The week of the year [1-48]. Use -1 for year-round.
        labels: List of all species labels
        threshold: Only values above or equal to threshold will be included
        sort: If the species list should be sorted alphabetically

    Returns:
        A list of species names that meet the threshold

    Raises:
        RuntimeError: If the predictor has not been initialized
    """
    if _predictor is None:
        raise RuntimeError(
            "Species predictor not initialized. Call init_species_predictor() first with the model path."
        )

    return _predictor.get_species_list(lat, lon, week, labels, threshold, sort)
