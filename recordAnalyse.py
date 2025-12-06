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
from config import BirdNetConfig


# Configure logging for Docker environment
def setup_logging() -> logging.Logger:
    """Configure logging to output to stdout for Docker container logs."""
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_format = os.environ.get(
        "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)


def confirm_dir(dir: Path) -> bool:
    if not dir.is_dir():
        os.mkdir(dir)
    return True


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
    config = BirdNetConfig.from_env(models_dir)

    logger.info(
        f"Configuration - Latitude: {config.latitude}, Longitude: {config.longitude}, Input Device: {config.input_device}"
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


def convert_frame_to_signal(frames: list[bytes]) -> np.ndarray:
    signal = np.frombuffer(b"".join(frames), dtype=np.int16)  # type: ignore
    return signal


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
                    (uuid7str(), start_ts, species_names[0], species_names[1], p[1])
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
    input_device: str = config.input_device
    record_seconds: int = 3
    period_size: int = 1024
    queue: asyncio.Queue[tuple[int, list[bytes]]] = asyncio.Queue()

    logger.info(f"Initializing audio input device: {input_device}")
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
