# pyright: reportMissingImports=false
import asyncio
import datetime
import os
import alsaaudio  # type: ignore
import numpy as np
import math
import operator
from typing import Any
from birdnet_analyzer.analyze.utils import load_codes, predict  # type: ignore
import birdnet_analyzer.utils as utils  # type: ignore
import birdnet_analyzer.config as cfg  # type: ignore
from birdnet_analyzer.species.utils import get_species_list  # type: ignore
import sqlite3
from zoneinfo import ZoneInfo
from uuid_extensions import uuid7str  # type: ignore
from pathlib import Path
import pydub  # type: ignore
import logging
import sys


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


def startup(w_dir: Path) -> None:
    logger.info("Starting up bird-listener application")
    recording_dir = w_dir / "recordings"
    _ = confirm_dir(recording_dir)
    logger.info(f"Recording directory: {recording_dir}")

    db_dir = w_dir / "db"
    _ = confirm_dir(db_dir)
    logger.info(f"Database directory: {db_dir}")

    script_dir = w_dir / "BirdNET-Analyzer"

    cfg.MODEL_PATH = os.path.join(script_dir, cfg.MODEL_PATH)
    cfg.LABELS_FILE = os.path.join(
        script_dir,
        "birdnet_analyzer/labels/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels_en_uk.txt",
    )
    cfg.TRANSLATED_LABELS_PATH = os.path.join(script_dir, cfg.TRANSLATED_LABELS_PATH)
    cfg.MDATA_MODEL_PATH = os.path.join(script_dir, cfg.MDATA_MODEL_PATH)
    cfg.CODES_FILE = os.path.join(script_dir, cfg.CODES_FILE)
    cfg.ERROR_LOG_FILE = os.path.join(script_dir, cfg.ERROR_LOG_FILE)
    cfg.TFLITE_THREADS = 1
    cfg.MIN_CONFIDENCE = 0.05
    cfg.SIGMOID_SENSITIVITY = 1.0
    cfg.CPU_THREADS = 1
    cfg.SAMPLE_RATE = 48000

    cfg.CODES = load_codes()
    cfg.LABELS = utils.read_lines(cfg.LABELS_FILE)
    cfg.TRANSLATED_LABELS = cfg.LABELS
    cfg.LOCATION_FILTER_THRESHOLD = 0.03
    cfg.WEEK = datetime.date.today().isocalendar()[
        1
    ]  # could change this to read from the recording file name

    # Read configuration from environment variables
    cfg.LATITUDE = float(os.environ.get("LATITUDE", "-10.0"))
    cfg.LONGITUDE = float(os.environ.get("LONGITUDE", "20.0"))
    cfg.INPUT_DEVICE = os.environ.get("INPUT_DEVICE_NAME", "default")
    logger.info(
        f"Configuration - Latitude: {cfg.LATITUDE}, Longitude: {cfg.LONGITUDE}, Input Device: {cfg.INPUT_DEVICE}"
    )

    cfg.SPECIES_LIST = get_species_list(
        cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK, cfg.LOCATION_FILTER_THRESHOLD
    )
    logger.info(f"Loaded {len(cfg.SPECIES_LIST)} species for location filtering")

    # add database configuration
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


def convert_frame_to_signal(frames: list[bytes]) -> np.ndarray:
    signal = np.frombuffer(b"".join(frames), dtype=np.int16)  # type: ignore
    return signal


def save_audio(
    species_name: str, signal: np.ndarray, start_ts: int, w_dir: Path
) -> bool:
    a = pydub.AudioSegment(  # type: ignore
        signal.tobytes(), frame_rate=cfg.SAMPLE_RATE, sample_width=2, channels=1
    )
    n = "ts_" + str(start_ts) + ".flac"
    d = w_dir / "recordings" / species_name
    _ = confirm_dir(d)
    fn = d / n
    a.export(fn, format="flac")  # type: ignore
    logger.debug(f"Saved audio file: {fn}")
    return True


def analyse_recording(start_ts: int, signal: np.ndarray, w_dir: Path) -> list[str]:
    # Convert standard signal to [-1,1] range
    scaled_sig: np.ndarray = signal / 32768

    # Run the analysis
    logger.debug("Running BirdNET analysis on audio signal")
    pred = predict([scaled_sig])[0]  # type: ignore

    # Assign scores to labels
    p_labels = zip(cfg.LABELS, pred)  # type: ignore

    # Sort by score
    p_sorted = sorted(p_labels, key=operator.itemgetter(1), reverse=True)  # type: ignore

    # Filter for scores above the confidence threshold
    predictions = list(filter(lambda x: (x[1] > cfg.MIN_CONFIDENCE), p_sorted))  # type: ignore

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
            if p[1] > cfg.MIN_CONFIDENCE and (
                not cfg.SPECIES_LIST or p[0] in cfg.SPECIES_LIST
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
    queue: asyncio.Queue[tuple[int, list[bytes]]], max_queue_size: int, w_dir: Path
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
            _ = analyse_recording(start_ts, signal, w_dir)

            # save the file
            # EXCLUSION_LIST = ['Trichoglossus moluccanus_Rainbow Lorikeet','Cacatua galerita_Sulphur-crested Cockatoo','Strepera graculina_Pied Currawong','Alisterus scapularis_Australian King-Parrot','Fulica atra_Eurasian Cootz','Gymnorhina tibicen_Australian Magpie']
            # if len(detections) == 1 and detections[0] not in EXCLUSION_LIST:
            #     save_audio(detections[0], signal, start_ts, w_dir)

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

    w_dir: Path = Path.cwd().parent
    startup(w_dir)
    max_queue_size: int = 6
    rate: int = cfg.SAMPLE_RATE  # type: ignore
    input_device: str = cfg.INPUT_DEVICE  # type: ignore
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
        asyncio.create_task(analysis_worker(queue, max_queue_size, w_dir)),
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
