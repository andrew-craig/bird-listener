"""Configuration management for bird-listener application.

This module provides a clean, type-safe configuration system that replaces
the dependency on birdnet_analyzer.config with a dataclass-based approach.
"""

import os
from dataclasses import dataclass, replace
from pathlib import Path


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
    def from_env(cls, models_dir: Path) -> "BirdNetConfig":
        """Create configuration from environment variables and models directory.

        Args:
            models_dir: Path to the models directory containing BirdNET models

        Returns:
            A new BirdNetConfig instance with values from environment variables
            or defaults
        """
        model_path = models_dir / "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
        labels_file = models_dir / "BirdNET_GLOBAL_6K_V2.4_Labels.txt"
        mdata_model_path = models_dir / "BirdNET_GLOBAL_6K_V2.4_MData_Model_FP16.tflite"
        error_log_file = models_dir / "error_log.txt"

        return cls(
            model_path=model_path,
            labels_file=labels_file,
            mdata_model_path=mdata_model_path,
            error_log_file=error_log_file,
            latitude=float(os.environ.get("LATITUDE", "-10.0")),
            longitude=float(os.environ.get("LONGITUDE", "20.0")),
            input_device=os.environ.get("INPUT_DEVICE_NAME", "default"),
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
            A new BirdNetConfig instance with runtime data populated
        """
        return replace(
            self,
            labels=tuple(labels),
            translated_labels=tuple(labels),  # Same as labels for now
            species_list=tuple(species_list),
        )
