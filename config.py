"""Configuration management for bird-listener application.

This module provides a clean, type-safe configuration system that replaces
the dependency on birdnet_analyzer.config with a dataclass-based approach.
"""

import os
from dataclasses import dataclass, replace
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


@dataclass(frozen=True)
class BirdNetConfig:
    """BirdNET analyzer configuration.

    This configuration is immutable (frozen=True) to prevent accidental
    modifications. Use dataclasses.replace() to create new instances
    with updated values.
    """

    # Model paths
    model_path: Path
    labels_file: Path
    mdata_model_path: Path
    error_log_file: Path

    # Audio settings
    sample_rate: int = 48000

    # Inference settings
    min_confidence: float = 0.05
    sigmoid_sensitivity: float = 1.0
    apply_sigmoid: bool = True
    tflite_threads: int = 1
    cpu_threads: int = 1

    # Location settings
    latitude: float = -10.0
    longitude: float = 20.0
    week: int = -1
    location_filter_threshold: float = 0.03

    # Device
    input_device: str = "default"

    # Runtime data (loaded during initialization)
    labels: tuple[str, ...] = ()
    translated_labels: tuple[str, ...] = ()
    species_list: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, models_dir: Path, logger=None) -> "BirdNetConfig":
        """Create configuration from environment variables and models directory.

        Args:
            models_dir: Path to the models directory containing BirdNET models
            logger: Optional logger for debugging configuration loading

        Returns:
            A new BirdNetConfig instance with values from environment variables
            or defaults
        """
        # Model paths
        model_path = models_dir / "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
        labels_file = models_dir / "BirdNET_GLOBAL_6K_V2.4_Labels.txt"
        mdata_model_path = models_dir / "BirdNET_GLOBAL_6K_V2.4_MData_Model_FP16.tflite"
        error_log_file = models_dir / "error_log.txt"

        # Check if environment variables are set (not just retrieving with defaults)
        latitude_is_set = "LATITUDE" in os.environ
        longitude_is_set = "LONGITUDE" in os.environ
        input_device_is_set = "INPUT_DEVICE_NAME" in os.environ

        # Read environment variables with explicit logging
        latitude_str = os.environ.get("LATITUDE", "-10.0")
        longitude_str = os.environ.get("LONGITUDE", "20.0")
        input_device = os.environ.get("INPUT_DEVICE_NAME", "default")

        if logger:
            if not latitude_is_set:
                logger.warning("LATITUDE environment variable not set, using default: -10.0")
            if not longitude_is_set:
                logger.warning("LONGITUDE environment variable not set, using default: 20.0")
            if not input_device_is_set:
                logger.info("INPUT_DEVICE_NAME not set, using default: 'default'")

        # Convert to appropriate types
        try:
            latitude = float(latitude_str)
        except ValueError:
            if logger:
                logger.error(f"Invalid LATITUDE value '{latitude_str}', using default -10.0")
            latitude = -10.0

        try:
            longitude = float(longitude_str)
        except ValueError:
            if logger:
                logger.error(f"Invalid LONGITUDE value '{longitude_str}', using default 20.0")
            longitude = 20.0

        # Log the loaded values for debugging
        if logger:
            logger.info(f"LATITUDE: {latitude_str} -> {latitude}")
            logger.info(f"LONGITUDE: {longitude_str} -> {longitude}")
            logger.info(f"INPUT_DEVICE_NAME: {input_device}")

        return cls(
            model_path=model_path,
            labels_file=labels_file,
            mdata_model_path=mdata_model_path,
            error_log_file=error_log_file,
            latitude=latitude,
            longitude=longitude,
            input_device=input_device,
        )

    def with_runtime_data(
        self,
        labels: list[str],
        species_list: list[str],
    ) -> "BirdNetConfig":
        """Create a new config instance with runtime data loaded.

        Args:
            labels: List of species labels loaded from the labels file
            species_list: List of species filtered by location and time

        Returns:
            A new Config instance with runtime data populated
        """
        return replace(
            self,
            labels=tuple(labels),
            translated_labels=tuple(labels),  # Same as labels for now
            species_list=tuple(species_list),
        )
