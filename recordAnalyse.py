# pyright: reportMissingImports=false
import asyncio
import datetime
import gc
import http.client
import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import alsaaudio  # type: ignore
import numpy as np
from uuid_extensions import uuid7str  # type: ignore

import getSpecies  # Geographic species filtering
import predict as tflite_predict  # Our lightweight TFLite predictor
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
        "taskName",
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

        if record.exc_info:
            log_data["exc_info"] = self.formatException(record.exc_info)

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

# BirdNET model expects exactly this many samples (3 seconds at 48kHz)
EXPECTED_SAMPLE_COUNT = 144000

# Max number of 3s chunks waiting for analysis. When full, the oldest chunk is
# dropped so memory stays bounded (~288KB per chunk) if analysis falls behind.
MAX_QUEUE_SIZE = 3

# How often to log a summary of activity at INFO level
SUMMARY_INTERVAL_SECONDS = 3600

# weather-server endpoint that receives each observation
WEATHER_SERVER_HOST = "127.0.0.1"
WEATHER_SERVER_PORT = 8000
WEATHER_SERVER_PATH = "/birds/latest"
WEATHER_SERVER_TIMEOUT_SECONDS = 5.0

DB_TABLE = "observations"


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


def startup(w_dir: Path) -> tuple[BirdNetConfig, getSpecies.WeeklySpeciesMasks]:
    """Initialize models and return configuration and the weekly species filter.

    Args:
        w_dir: Working directory path

    Returns:
        BirdNetConfig instance with labels loaded, and the per-week species masks
    """
    logger.info("Starting up bird-listener application")
    models_dir = w_dir / "models"

    # Create base configuration from environment
    logger.info("Loading configuration from environment variables")
    config = BirdNetConfig.from_env(models_dir, logger=logger)

    logger.info(
        f"Configuration loaded - Latitude: {config.latitude}, Longitude: {config.longitude}, Input Device: {config.input_device}"
    )

    # Load labels from file
    labels = config.labels_file.read_text(encoding="utf-8").splitlines()

    # The metadata model is only needed at startup: compute the location
    # filter for all 48 BirdNET weeks now, then free the model before loading
    # the (much larger) main model. The analyser picks the mask for the
    # current date, so the filter follows the season without reloading.
    logger.info(f"Initializing species predictor with model: {config.mdata_model_path}")
    getSpecies.init_species_predictor(str(config.mdata_model_path), config.tflite_threads)
    weekly_masks = getSpecies.build_weekly_masks(
        config.latitude, config.longitude, len(labels), config.location_filter_threshold
    )
    getSpecies.release_species_predictor()
    gc.collect()
    logger.info("Computed location species filter for all 48 weeks")

    # Initialize the TFLite predictor with the model
    logger.info(f"Initializing TFLite predictor with model: {config.model_path}")
    tflite_predict.init_predictor(
        str(config.model_path), config.tflite_threads, EXPECTED_SAMPLE_COUNT
    )
    logger.info("TFLite predictor initialized successfully")

    # Return config with runtime data loaded
    return config.with_runtime_data(labels), weekly_masks


@dataclass
class Stats:
    """Counters for the periodic INFO summary."""

    chunks_analysed: int = 0
    chunks_dropped: int = 0
    capture_overruns: int = 0
    detections: int = 0
    post_failures: int = 0
    inference_seconds: float = 0.0

    def log_and_reset(self, interval_seconds: float) -> None:
        avg_ms = (
            1000 * self.inference_seconds / self.chunks_analysed
            if self.chunks_analysed
            else 0.0
        )
        logger.info(
            "Summary: %d chunks analysed, %d dropped, %d capture overruns, "
            "%d detections, %d failed posts, avg inference %.0f ms",
            self.chunks_analysed,
            self.chunks_dropped,
            self.capture_overruns,
            self.detections,
            self.post_failures,
            avg_ms,
            extra={"interval_seconds": round(interval_seconds)},
        )
        self.chunks_analysed = 0
        self.chunks_dropped = 0
        self.capture_overruns = 0
        self.detections = 0
        self.post_failures = 0
        self.inference_seconds = 0.0


class AudioChunker:
    """Reads fixed-size chunks of exactly ``chunk_samples`` from an ALSA PCM.

    Each period is copied straight into a preallocated int16 array, so there
    is no list of byte strings to join and no padding/trimming afterwards.
    Samples from a period that straddles a chunk boundary are carried over
    to the start of the next chunk, so no audio is lost between chunks.
    """

    def __init__(self, pcm: Any, stats: Stats, chunk_samples: int = EXPECTED_SAMPLE_COUNT):
        self.pcm = pcm
        self.stats = stats
        self.chunk_samples = chunk_samples
        self._carry: np.ndarray = np.empty(0, dtype=np.int16)

    def read_chunk(self) -> tuple[int, np.ndarray]:
        """Block until a full chunk has been captured. Run this in a thread."""
        start_ts = int(time.time())
        chunk = np.empty(self.chunk_samples, dtype=np.int16)
        filled = len(self._carry)
        chunk[:filled] = self._carry
        self._carry = np.empty(0, dtype=np.int16)

        while filled < self.chunk_samples:
            length, data = self.pcm.read()  # type: ignore
            if length <= 0:
                # Negative length means an overrun (-EPIPE); pyalsaaudio has
                # already re-prepared the stream, so just keep reading.
                if length < 0:
                    self.stats.capture_overruns += 1
                    logger.debug("ALSA capture overrun (%d)", length)
                continue
            samples = np.frombuffer(data, dtype=np.int16)
            take = min(len(samples), self.chunk_samples - filled)
            chunk[filled : filled + take] = samples[:take]
            filled += take
            if take < len(samples):
                self._carry = samples[take:].copy()

        return start_ts, chunk


class WeatherServerClient:
    """Posts observations to weather-server over a reused keep-alive connection.

    Uses the standard library's http.client rather than ``requests``, which
    saves ~10MB of resident memory, and always applies a timeout.
    """

    def __init__(self, host: str, port: int, path: str, timeout: float):
        self.host = host
        self.port = port
        self.path = path
        self.timeout = timeout
        self._conn: http.client.HTTPConnection | None = None

    def _close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def post(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        # One retry, only when a reused keep-alive connection turns out to be
        # stale (the server closed it while idle).
        for attempt in range(2):
            reused = self._conn is not None
            if self._conn is None:
                self._conn = http.client.HTTPConnection(
                    self.host, self.port, timeout=self.timeout
                )
            try:
                self._conn.request("POST", self.path, body, headers)
                response = self._conn.getresponse()
                response.read()
                if response.will_close:
                    self._close()
                if response.status >= 400:
                    raise RuntimeError(f"weather-server returned HTTP {response.status}")
                return
            except (
                http.client.RemoteDisconnected,
                BrokenPipeError,
                ConnectionResetError,
            ):
                self._close()
                if not reused or attempt == 1:
                    raise
            except Exception:
                self._close()
                raise


class Analyser:
    """Runs inference on audio chunks and stores/forwards detections.

    Holds long-lived resources (SQLite connection, HTTP connection, a reusable
    float32 input buffer and a precomputed species mask) so nothing heavy is
    created per chunk. ``process`` is called from a worker thread, one chunk
    at a time.
    """

    def __init__(
        self,
        config: BirdNetConfig,
        weekly_masks: getSpecies.WeeklySpeciesMasks,
        db_path: Path,
        stats: Stats,
    ):
        self.config = config
        self.weekly_masks = weekly_masks
        self.stats = stats
        self.labels = config.labels

        # Reused for every chunk: int16 -> float32 [-1, 1] without temporaries
        self._input = np.empty(EXPECTED_SAMPLE_COUNT, dtype=np.float32)
        self._scale = np.float32(1.0 / 32768.0)

        logger.info(f"Initializing database: {db_path}")
        # Only ever used by one analysis call at a time, but that call may run
        # on any thread of the default executor.
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute(
            f"CREATE TABLE IF NOT EXISTS {DB_TABLE} "
            "(id text, ts integer, scientific_name txt, common_name txt, confidence real)"
        )
        self.db.commit()
        logger.info("Database initialized successfully")

        self.weather_server = WeatherServerClient(
            WEATHER_SERVER_HOST,
            WEATHER_SERVER_PORT,
            WEATHER_SERVER_PATH,
            WEATHER_SERVER_TIMEOUT_SECONDS,
        )

    def close(self) -> None:
        self.db.close()

    def process(self, start_ts: int, signal: np.ndarray) -> list[str]:
        """Analyse one chunk; returns the labels of stored detections."""
        np.multiply(signal, self._scale, out=self._input, dtype=np.float32)

        t0 = time.perf_counter()
        pred = tflite_predict.predict(self._input)
        self.stats.inference_seconds += time.perf_counter() - t0
        self.stats.chunks_analysed += 1

        if self.config.apply_sigmoid:
            pred = tflite_predict.flat_sigmoid(
                pred, sensitivity=-1, bias=self.config.sigmoid_sensitivity
            )

        # Location/season filter for the chunk's local date (cached per week)
        week, allowed_mask, changed = self.weekly_masks.for_date(
            datetime.date.fromtimestamp(start_ts)
        )
        if changed:
            logger.info(
                "Species filter set for BirdNET week %d: %d species",
                week,
                int(allowed_mask.sum()),
            )

        # Only the few labels above threshold (and allowed for this location)
        # are pulled out and sorted, instead of sorting all ~6.5k labels.
        idx = np.flatnonzero((pred > self.config.min_confidence) & allowed_mask)
        if idx.size == 0:
            logger.debug("No detections above confidence threshold")
            return []
        idx = idx[np.argsort(pred[idx])[::-1]]

        rows: list[tuple[str, int, str, str, float]] = []
        for i in idx:
            scientific_name, common_name = self.labels[i].split("_", 1)
            rows.append((uuid7str(), start_ts, scientific_name, common_name, float(pred[i])))

        self.db.executemany(f"INSERT INTO {DB_TABLE} VALUES(?, ?, ?, ?, ?)", rows)
        self.db.commit()
        self.stats.detections += len(rows)

        logger.info(
            "Detected %s",
            ", ".join(f"{r[3]} ({r[4]:.2f})" for r in rows),
            extra={
                "detections": [
                    {"common_name": r[3], "confidence": round(r[4], 3)} for r in rows
                ]
            },
        )

        for ob in rows:
            payload = {
                "id": ob[0],
                "ts": ob[1],
                "scientific_name": ob[2],
                "common_name": ob[3],
                "confidence": ob[4],
            }
            try:
                self.weather_server.post(payload)
                logger.debug("Sent observation %s to weather-server", ob[0])
            except Exception as e:
                self.stats.post_failures += 1
                logger.error(f"Failed to send data to weather-server: {e}")

        return [self.labels[i] for i in idx]


async def recording_worker(
    queue: asyncio.Queue[tuple[int, np.ndarray]],
    chunker: AudioChunker,
    stats: Stats,
) -> None:
    while True:
        # Blocking ALSA reads run in a thread, so the event loop is not woken
        # per period and there are no sleep() gaps that could overrun ALSA.
        item = await asyncio.to_thread(chunker.read_chunk)

        if queue.full():
            # Analysis is behind: shed the oldest chunk so memory stays bounded
            # and we keep analysing the most recent audio.
            queue.get_nowait()
            stats.chunks_dropped += 1
            logger.warning("Analysis queue full (%d), dropped oldest chunk", queue.maxsize)
        queue.put_nowait(item)
        logger.debug("Queued chunk (queue size: %d)", queue.qsize())


async def analysis_worker(
    queue: asyncio.Queue[tuple[int, np.ndarray]],
    analyser: Analyser,
) -> None:
    while True:
        start_ts, signal = await queue.get()
        try:
            # Inference blocks for a while; keep it off the event loop so the
            # recorder keeps capturing in the meantime.
            await asyncio.to_thread(analyser.process, start_ts, signal)
        except Exception:
            logger.exception("Analysis failed for chunk starting at %d", start_ts)
        finally:
            del signal


async def summary_worker(stats: Stats, interval: float = SUMMARY_INTERVAL_SECONDS) -> None:
    while True:
        await asyncio.sleep(interval)
        stats.log_and_reset(interval)


async def main() -> None:
    # Initialize logging first
    _ = setup_logging()
    logger.info("Bird-listener application starting")

    w_dir: Path = Path.cwd()
    db_dir = w_dir / "db"
    _ = confirm_dir(db_dir)

    config, weekly_masks = startup(w_dir)
    stats = Stats()
    analyser = Analyser(config, weekly_masks, db_dir / "bird-observations.db", stats)

    rate: int = config.sample_rate
    period_size: int = 1024
    queue: asyncio.Queue[tuple[int, np.ndarray]] = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)

    # Find a working capture device
    logger.info(f"Preferred audio input device from config: {config.input_device}")
    try:
        input_device = find_working_capture_device(config.input_device)
        logger.info(f"Using capture device: {input_device}")
    except RuntimeError as e:
        logger.error(f"Failed to find working capture device: {e}")
        raise

    logger.info(f"Sample rate: {rate} Hz, chunk: {EXPECTED_SAMPLE_COUNT} samples")

    pcm = alsaaudio.PCM(  # type: ignore
        alsaaudio.PCM_CAPTURE,
        alsaaudio.PCM_NORMAL,
        channels=1,
        rate=rate,
        format=alsaaudio.PCM_FORMAT_S16_LE,
        periodsize=period_size,
        device=input_device,
    )
    chunker = AudioChunker(pcm, stats)

    logger.info("Starting recording and analysis workers")

    try:
        _ = await asyncio.gather(
            analysis_worker(queue, analyser),
            recording_worker(queue, chunker, stats),
            summary_worker(stats),
        )
    finally:
        logger.info("Shutting down audio stream")
        pcm.close()  # type: ignore
        analyser.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Application stopped by user")
    except Exception as e:
        logging.getLogger(__name__).error(f"Application error: {e}", exc_info=True)
        raise
