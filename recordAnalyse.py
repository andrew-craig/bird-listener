# pyright: reportMissingImports=false
import asyncio
import datetime
import os
import alsaaudio  # type: ignore
import numpy as np
import math
import operator
from typing import Any
import predict as tflite_predict  # Our lightweight TFLite predictor
import getSpecies  # Geographic species filtering
import sqlite3
from zoneinfo import ZoneInfo
from uuid_extensions import uuid7str  # type: ignore
from pathlib import Path
import pydub  # type: ignore
import logging
import sys
import requests
import json
from config import BirdNetConfig


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    # Standard LogRecord attributes to exclude from extra fields
    RESERVED_ATTRS = {
        "name",
        "msg",
        "args",
        "created",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "thread",
        "threadName",
        "exc_info",
        "exc_text",
        "stack_info",
        "asctime",
    }

    def formatTime(self, record, datefmt=None):
        """Format timestamp in RFC 3339 / ISO 8601 format."""
        dt = datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(record.msecs):03d}Z"

    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "service": "bird-listener",
        }

        # Add any custom fields from extra parameter
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS:
                log_data[key] = value

        return json.dumps(log_data)


# Configure logging for Docker environment
def setup_logging() -> logging.Logger:
    """Configure logging to output to stdout for Docker container logs."""
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        handlers=[handler],
        force=True,
    )

    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)


def confirm_dir(dir: Path) -> bool:
    if not dir.is_dir():
        os.mkdir(dir)
    return True


def find_working_capture_device(preferred_device: str) -> str:
    """Find a working ALSA capture device.

    Args:
        preferred_device: Device name from configuration to try first

    Returns:
        Name of a working capture device

    Raises:
        RuntimeError: If no working capture device is found
    """
    # Get list of available PCM devices
    try:
        pcm_list = alsaaudio.pcms(alsaaudio.PCM_CAPTURE)  # type: ignore
        logger.info(f"Available ALSA capture devices: {pcm_list}")
    except Exception as e:
        logger.error(f"Failed to list ALSA capture devices: {e}")
        pcm_list = []

    # Try the preferred device first if it's in the list
    devices_to_try = []
    if preferred_device in pcm_list:
        devices_to_try.append(preferred_device)

    # Add all other devices
    devices_to_try.extend([d for d in pcm_list if d != preferred_device])

    # If preferred device wasn't in the list but isn't "default", still try it
    if preferred_device not in pcm_list and preferred_device != "default":
        devices_to_try.insert(0, preferred_device)

    # Try each device
    for device in devices_to_try:
        logger.info(f"Testing capture device: {device}")
        try:
            # Try to open the device with minimal configuration
            test_pcm = alsaaudio.PCM(  # type: ignore
                alsaaudio.PCM_CAPTURE,
                alsaaudio.PCM_NORMAL,
                channels=1,
                rate=48000,
                format=alsaaudio.PCM_FORMAT_S16_LE,
                periodsize=1024,
                device=device,
            )
            test_pcm.close()  # type: ignore
            logger.info(f"Successfully opened capture device: {device}")
            return device
        except alsaaudio.ALSAAudioError as e:  # type: ignore
            logger.warning(f"Device '{device}' failed to open: {e}")
            continue

    # If we get here, no device worked
    error_msg = f"No working ALSA capture device found. Tried: {devices_to_try}"
    logger.error(error_msg)
    raise RuntimeError(error_msg)


def startup(w_dir: Path) -> BirdNetConfig:
    """Initialize the application and return configuration.

    Args:
        w_dir: Working directory path

    Returns:
        BirdNetConfig instance with all runtime data loaded
    """
    logger.info("Starting up bird-listener application")
    recording_dir = w_dir / "recordings"
    _ = confirm_dir(recording_dir)
    logger.info(f"Recording directory: {recording_dir}")

    db_dir = w_dir / "db"
    _ = confirm_dir(db_dir)
    logger.info(f"Database directory: {db_dir}")

    models_dir = w_dir / "models"

    # Create base configuration from environment
    logger.info("Loading configuration from environment variables")
    config = BirdNetConfig.from_env(models_dir, logger=logger)

    logger.info(
        f"Configuration loaded - Latitude: {config.latitude}, Longitude: {config.longitude}, Input Device: {config.input_device}"
    )

    # Load labels from file
    labels = config.labels_file.read_text(encoding="utf-8").splitlines()

    # Calculate current week for species filtering
    week = datetime.date.today().isocalendar()[1]

    # Initialize species predictor for geographic filtering
    logger.info(f"Initializing species predictor with model: {config.mdata_model_path}")
    getSpecies.init_species_predictor(str(config.mdata_model_path), config.tflite_threads)

    # Get filtered species list for location
    species_list = getSpecies.get_species_list(
        config.latitude, config.longitude, week, labels, config.location_filter_threshold
    )
    logger.info(f"Loaded {len(species_list)} species for location filtering")

    # Initialize the TFLite predictor with the model
    logger.info(f"Initializing TFLite predictor with model: {config.model_path}")
    tflite_predict.init_predictor(str(config.model_path), config.tflite_threads)
    logger.info("TFLite predictor initialized successfully")

    # Initialize database
    database = db_dir / "bird-observations.db"
    table_name = "observations"
    columns = (
        "id text, ts integer, scientific_name txt, common_name txt, confidence real"
    )
    logger.info(f"Initializing database: {database}")
    connection = sqlite3.connect(database)
    cursor = connection.cursor()
    _ = cursor.execute(
        """CREATE TABLE IF NOT EXISTS {table_name} ({columns})""".format(
            table_name=table_name, columns=columns
        )
    )
    connection.commit()
    connection.close()
    logger.info("Database initialized successfully")

    # Return config with runtime data loaded
    return config.with_runtime_data(labels, species_list)


# BirdNET model expects exactly this many samples (3 seconds at 48kHz)
EXPECTED_SAMPLE_COUNT = 144000


def convert_frame_to_signal(frames: list[bytes]) -> np.ndarray:
    signal = np.frombuffer(b"".join(frames), dtype=np.int16)  # type: ignore
    return signal


def normalize_signal_length(signal: np.ndarray, expected_length: int = EXPECTED_SAMPLE_COUNT) -> np.ndarray:
    """Pad or trim audio signal to exact expected length for TFLite model.

    The BirdNET TFLite model requires exactly 144,000 samples. ALSA period-based
    recording may produce slightly more or fewer samples due to integer rounding.
    This function ensures the signal matches the expected length exactly.

    Args:
        signal: Audio signal as numpy array
        expected_length: Expected number of samples (default: 144000)

    Returns:
        Signal padded with zeros or trimmed to exact expected length
    """
    current_length = len(signal)
    if current_length == expected_length:
        return signal
    elif current_length < expected_length:
        # Pad with zeros at the end
        padding = expected_length - current_length
        logger.debug(f"Padding signal with {padding} zeros ({current_length} -> {expected_length})")
        return np.pad(signal, (0, padding), mode='constant', constant_values=0)
    else:
        # Trim excess samples from the end
        logger.debug(f"Trimming signal by {current_length - expected_length} samples ({current_length} -> {expected_length})")
        return signal[:expected_length]


def save_audio(
    species_name: str, signal: np.ndarray, start_ts: int, w_dir: Path, config: BirdNetConfig
) -> bool:
    a = pydub.AudioSegment(  # type: ignore
        signal.tobytes(), frame_rate=config.sample_rate, sample_width=2, channels=1
    )
    n = "ts_" + str(start_ts) + ".flac"
    d = w_dir / "recordings" / species_name
    _ = confirm_dir(d)
    fn = d / n
    a.export(fn, format="flac")  # type: ignore
    logger.debug(f"Saved audio file: {fn}")
    return True


def analyse_recording(start_ts: int, signal: np.ndarray, w_dir: Path, config: BirdNetConfig) -> list[str]:
    # Ensure signal is exactly the length the model expects
    signal = normalize_signal_length(signal)

    # Convert standard signal to [-1,1] range
    scaled_sig: np.ndarray = signal / 32768

    # Run the analysis using our lightweight TFLite predictor
    logger.debug("Running BirdNET analysis on audio signal")
    pred = tflite_predict.predict([scaled_sig])[0]  # type: ignore

    # Apply sigmoid if configured
    if config.apply_sigmoid:
        pred = tflite_predict.flat_sigmoid(
            np.array(pred), sensitivity=-1, bias=config.sigmoid_sensitivity
        )

    # Assign scores to labels
    p_labels = zip(config.labels, pred)

    # Sort by score
    p_sorted = sorted(p_labels, key=operator.itemgetter(1), reverse=True)  # type: ignore

    # Filter for scores above the confidence threshold
    predictions = list(filter(lambda x: (x[1] > config.min_confidence), p_sorted))

    # SAVE THE RESULTS
    if len(predictions) == 0:
        logger.debug("No detections above confidence threshold")
        return []
    else:
        # pull the timestamp from the file name
        logger.info(f"Detected {len(predictions)} sounds above confidence threshold")

        ## save to the db
        database = w_dir / "db" / "bird-observations.db"
        table_name = "observations"
        connection = sqlite3.connect(database)
        cursor = connection.cursor()

        # reformat the data
        data_to_insert: list[tuple[str, int, str, str, float]] = []
        filtered_detections: list[str] = []
        for p in predictions:
            if p[1] > config.min_confidence and (
                not config.species_list or p[0] in config.species_list
            ):
                species_names = p[0].split("_")
                logger.info(f"{species_names[1]} detected (confidence: {p[1]:.2f})")
                data_to_insert.append(
                    (uuid7str(), start_ts, species_names[0], species_names[1], float(p[1]))
                )
                filtered_detections.append(p[0])

        if len(data_to_insert) > 0:
            # insert to table and commit change
            _ = cursor.executemany(
                "INSERT INTO {table_name} VALUES(?, ?, ?, ?, ?)".format(
                    table_name=table_name
                ),
                data_to_insert,
            )
            connection.commit()
            logger.info(f"Inserted {len(data_to_insert)} detections into database")

            # close the connection
            connection.close()

            # Call weather-server to save
            for ob in data_to_insert:
                try:
                    payload = {
                        "id": ob[0],
                        "ts": ob[1],
                        "scientific_name": ob[2],
                        "common_name": ob[3],
                        "confidence": ob[4],
                    }
                    # Send to weather-server
                    r = requests.post(
                        "http://127.0.0.1:8000/birds/latest", json=payload
                    )
                except Exception as e:
                    logger.error(f"Failed to send data to weather-server: {e}")

        return filtered_detections


async def recording_worker(
    queue: asyncio.Queue[tuple[int, list[bytes]]],
    input_device: Any,  # alsaaudio.PCM object
    num_periods: int = 140,
) -> None:
    while True:
        start_ts: int = int(datetime.datetime.now(tz=ZoneInfo("UTC")).timestamp())
        frames: list[bytes] = []
        for _ in range(num_periods):
            _length, data = input_device.read()  # type: ignore
            frames.append(data)
            await asyncio.sleep(0.01)

        logger.debug(f"Completed recording loop, queue size: {queue.qsize()}")
        await queue.put((start_ts, frames))


async def analysis_worker(
    queue: asyncio.Queue[tuple[int, list[bytes]]], max_queue_size: int, w_dir: Path, config: BirdNetConfig
) -> None:
    while True:
        logger.debug("Analysis worker waiting for queue item")
        # Get a "work item" out of the queue.
        start_ts: int
        frames: list[bytes]
        start_ts, frames = await queue.get()
        logger.debug(f"Picked up item from queue (queue size: {queue.qsize()})")
        if queue.qsize() <= max_queue_size:
            # analyse the file
            signal: np.ndarray = convert_frame_to_signal(frames)
            _ = analyse_recording(start_ts, signal, w_dir, config)

            # save the file
            # EXCLUSION_LIST = ['Trichoglossus moluccanus_Rainbow Lorikeet','Cacatua galerita_Sulphur-crested Cockatoo','Strepera graculina_Pied Currawong','Alisterus scapularis_Australian King-Parrot','Fulica atra_Eurasian Cootz','Gymnorhina tibicen_Australian Magpie']
            # if len(detections) == 1 and detections[0] not in EXCLUSION_LIST:
            #     save_audio(detections[0], signal, start_ts, w_dir, config)

            logger.debug("Analysis completed")
        else:
            logger.warning(
                f"Queue size ({queue.qsize()}) exceeds max ({max_queue_size}), skipping analysis"
            )

        await asyncio.sleep(0.5)
        # Notify the queue that the "work item" has been processed.
        queue.task_done()


async def main() -> None:
    # Initialize logging first
    _ = setup_logging()
    logger.info("Bird-listener application starting")

    w_dir: Path = Path.cwd()
    config = startup(w_dir)
    max_queue_size: int = 6
    rate: int = config.sample_rate
    record_seconds: int = 3
    period_size: int = 1024
    queue: asyncio.Queue[tuple[int, list[bytes]]] = asyncio.Queue()

    # Find a working capture device
    logger.info(f"Preferred audio input device from config: {config.input_device}")
    try:
        input_device = find_working_capture_device(config.input_device)
        logger.info(f"Using capture device: {input_device}")
    except RuntimeError as e:
        logger.error(f"Failed to find working capture device: {e}")
        raise

    logger.info(f"Sample rate: {rate} Hz, Record duration: {record_seconds}s")

    input = alsaaudio.PCM(  # type: ignore
        alsaaudio.PCM_CAPTURE,
        alsaaudio.PCM_NORMAL,
        channels=1,
        rate=rate,
        format=alsaaudio.PCM_FORMAT_S16_LE,
        periodsize=period_size,
        device=input_device,
    )
    period_size = input.info().get("period_size", period_size)  # type: ignore
    s: int = math.floor(record_seconds * (rate / period_size))

    logger.info("Starting recording and analysis workers")

    _ = await asyncio.gather(
        asyncio.create_task(analysis_worker(queue, max_queue_size, w_dir, config)),
        recording_worker(queue, input, s),
    )

    # Shut the stream
    logger.info("Shutting down audio stream")
    input.close()  # type: ignore


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Application stopped by user")
    except Exception as e:
        logging.getLogger(__name__).error(f"Application error: {e}", exc_info=True)
        raise
