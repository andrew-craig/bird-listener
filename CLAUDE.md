# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A lightweight bird detection system adapted from [BirdNET-Analyzer](https://github.com/birdnet-team/BirdNET-Analyzer) for deployment on resource-constrained devices like Raspberry Pi Zero 2 W. The system continuously records audio from a microphone, analyzes it using TFLite models to detect bird species, and stores observations in a SQLite database.

## Environment Setup

### Initial Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `source venv/bin/activate` if using venv

# Install Python dependencies
pip install -r requirements.txt
```

### Configuration
Copy `.env.template` to `.env` and configure:
- `TZ`: Timezone (e.g., "UTC")
- `LATITUDE` and `LONGITUDE`: Location coordinates for species filtering
- `INPUT_DEVICE_NAME`: ALSA audio input device (default: "default")
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.). At INFO, only detections, warnings/errors and an hourly summary are logged
- `MIN_CONFIDENCE`: Detection threshold (default 0.05; BirdNET itself typically uses 0.25-0.5)
- `TFLITE_THREADS`: CPU threads for inference (default 1)

### Model Installation
Models are not included in the repository. Download BirdNET V2.4 models:
```bash
mkdir models
cd models
wget https://drive.google.com/file/d/1ixYBPbZK2Fh1niUQzadE2IWTFZlwATa3
unzip -q V2.4.zip
mv V2.4/* .
rmdir V2.4
```

Required model files in `models/`:
- `BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite` - Main classification model
- `BirdNET_GLOBAL_6K_V2.4_Labels.txt` - Species labels
- `BirdNET_GLOBAL_6K_V2.4_MData_Model_FP16.tflite` - Geographic/temporal filtering model

## Running the Application

### Development Mode
```bash
# From repository root
python recordAnalyse.py
```

### Production Deployment (systemd)
Create `/etc/systemd/system/bird-listener.service`:
```ini
[Unit]
Description=Service to listen for birds via a mic
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
User=operator
Group=operator
SupplementaryGroups=audio
ExecStart=/home/operator/bird-listener/venv/bin/python /home/operator/bird-listener/recordAnalyse.py
WorkingDirectory=/home/operator/bird-listener
Restart=on-failure
RestartSec=5s
# Fewer glibc malloc arenas: less memory fragmentation from worker threads
Environment=MALLOC_ARENA_MAX=2
# Hardening: a mic listener needs no root and only writes to db/
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=/home/operator/bird-listener/db
PrivateTmp=true
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true

[Install]
WantedBy=multi-user.target
```

`ReadWritePaths` must exist before the unit starts, and the service user needs to own it and be in the `audio` group:
```bash
mkdir -p /home/operator/bird-listener/db
sudo chown -R operator:operator /home/operator/bird-listener/db  # was root-owned if the service ran as root
sudo usermod -aG audio operator
```

Then:
```bash
sudo systemctl enable bird-listener.service
sudo systemctl start bird-listener.service
```

## Architecture

### Core Components

**recordAnalyse.py** - Main application entry point
- Implements async producer-consumer pattern with a bounded queue (`MAX_QUEUE_SIZE`):
  - `recording_worker`: `AudioChunker` reads exactly 144,000 samples per chunk from ALSA in a thread (`asyncio.to_thread`); when the queue is full the oldest chunk is dropped
  - `analysis_worker`: runs `Analyser.process` in a thread so capture continues during inference
  - `summary_worker`: logs an hourly INFO summary (chunks analysed/dropped, overruns, detections, avg inference time)
- On startup, builds the location-filtered species list with the metadata model and then frees that model, loads the main model, and opens one long-lived SQLite connection
- Detections are stored in `db/bird-observations.db` and sent to weather-server (`http.client` keep-alive connection, 5s timeout)

**config.py** - Configuration management
- `BirdNetConfig` dataclass: Immutable configuration loaded from environment variables
- `from_env()`: Creates config from environment with sensible defaults
- `with_runtime_data()`: Adds labels and species list after initialization

**predict.py** - TFLite audio classification
- `TFLitePredictor`: Wraps BirdNET classification model inference
- Tensors are allocated once for a fixed `(1, 144000)` input; `predict()` takes a single float32 sample
- Global singleton pattern via `init_predictor()` and `predict()` functions
- `flat_sigmoid()`: Applies sigmoid activation for probability calibration

**getSpecies.py** - Geographic/temporal species filtering
- `SpeciesPredictor`: Uses metadata model to filter species by location and week
- Reduces false positives by limiting predictions to regionally/seasonally appropriate species
- Global singleton pattern similar to predict.py; `release_species_predictor()` frees it after startup

### Data Flow

1. Audio capture: ALSA → exactly 144,000-sample (3s) int16 chunks at 48kHz
2. Queue management: Recording worker pushes to a bounded queue, dropping the oldest chunk when full
3. Analysis: Consumer pulls from queue, converts to float32 [-1, 1] in a reused buffer
4. Classification: Main model predicts species from audio
5. Post-processing: Apply sigmoid if configured; keep labels above threshold that are in the location-filtered species list (precomputed boolean mask)
6. Storage: Insert detections to SQLite, POST each to weather-server

### Key Configuration Parameters

- `sample_rate`: 48000 Hz (BirdNET requirement)
- `min_confidence`: 0.05 (threshold for detections, `MIN_CONFIDENCE`)
- `sigmoid_sensitivity`: 1.0 (calibration parameter)
- `location_filter_threshold`: 0.03 (species filtering threshold)
- `tflite_threads`: 1 (CPU threads for inference, `TFLITE_THREADS`)

### BirdNET Analyzer Directory

The `birdnet_analyzer/` directory contains the original BirdNET-Analyzer codebase, which this project formerly depended on. This project has been refactored to remove dependencies on that code in favor of lightweight TFLite-only implementations. The directory remains for reference but is not used at runtime.

## Database Schema

SQLite database at `db/bird-observations.db`:

**observations** table:
- `id` (text): UUID7 identifier
- `ts` (integer): Unix timestamp
- `scientific_name` (text): Species scientific name
- `common_name` (text): Species common name
- `confidence` (real): Detection confidence score

## Audio Configuration

Verify ALSA audio device configuration:
```bash
alsamixer  # Check default mic and levels
arecord -l  # List capture devices
```

The system expects 16-bit signed little-endian PCM audio at 48kHz mono.
