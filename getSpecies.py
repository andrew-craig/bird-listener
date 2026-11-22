"""
Species list prediction module for BirdNET.

This module provides geographic and temporal filtering of bird species
using the BirdNET metadata model. It predicts which species are likely
to be present at a given location and time of year.
"""

import datetime
import math

import numpy as np

# The metadata model's notion of a week: 4 per month, 1-48
WEEKS_PER_YEAR = 48


def birdnet_week(date: datetime.date) -> int:
    """Convert a date to the metadata model's week number (1-48, 4 per month).

    This is not the ISO calendar week: BirdNET splits every month into four
    "weeks" (days 1-7, 8-14, 15-21, 22-end), matching BirdNET-Analyzer.
    """
    return (date.month - 1) * 4 + min(4, math.ceil(date.day / 7))

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


class WeeklySpeciesMasks:
    """Location filter for every week of the year, computed once at startup.

    Holds one boolean "species allowed" mask per BirdNET week, bit-packed
    (48 x ~6.5k labels is ~40KB), so the metadata model can be freed after
    startup while the filter still follows the time of year.
    """

    def __init__(self, packed: np.ndarray, num_labels: int):
        self._packed = packed
        self._num_labels = num_labels
        self._cached_week: int | None = None
        self._cached_mask: np.ndarray | None = None

    def mask_for_week(self, week: int) -> np.ndarray:
        """Boolean mask over labels for a week (1-48). All True if no species pass."""
        mask = np.unpackbits(self._packed[week - 1], count=self._num_labels).astype(bool)
        if not mask.any():
            # Nothing passes the threshold for this location/week: don't
            # filter at all rather than silently dropping every detection.
            mask[:] = True
        return mask

    def for_date(self, date: datetime.date) -> tuple[int, np.ndarray, bool]:
        """Mask for the given date.

        Returns (week, mask, changed) where changed is True when the week
        differs from the previous call (the mask is cached between calls).
        """
        week = birdnet_week(date)
        changed = week != self._cached_week
        if changed:
            self._cached_week = week
            self._cached_mask = self.mask_for_week(week)
        return week, self._cached_mask, changed  # type: ignore[return-value]


def build_weekly_masks(
    lat: float, lon: float, num_labels: int, threshold: float
) -> WeeklySpeciesMasks:
    """Run the metadata model for all 48 weeks using the global predictor."""
    if _predictor is None:
        raise RuntimeError(
            "Species predictor not initialized. Call init_species_predictor() first with the model path."
        )
    masks = np.zeros((WEEKS_PER_YEAR, num_labels), dtype=bool)
    for week in range(1, WEEKS_PER_YEAR + 1):
        scores = _predictor.predict_filter(lat, lon, week)
        if len(scores) != num_labels:
            raise ValueError(
                f"Metadata model returned {len(scores)} scores but there are {num_labels} labels"
            )
        masks[week - 1] = scores >= threshold
    return WeeklySpeciesMasks(np.packbits(masks, axis=1), num_labels)


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


def release_species_predictor() -> None:
    """Drop the global species predictor so its interpreter memory can be freed.

    The metadata model is only needed once at startup to build the species
    list, so there is no reason to keep it resident for the life of the process.
    """
    global _predictor
    _predictor = None
