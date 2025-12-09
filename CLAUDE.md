# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A lightweight bird detection system adapted from [BirdNET-Analyzer](https://github.com/birdnet-team/BirdNET-Analyzer) for deployment on resource-constrained devices like Raspberry Pi Zero 2 W. The system continuously records audio from a microphone, analyzes it using TFLite models to detect bird species, and stores observations in a SQLite database.

## Environment Setup

### Initial Setup
```bash
# Install system dependencies
sudo apt-get install ffmpeg

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
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)

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
ExecStart=/home/operator/bird-listener/venv/bin/python /home/operator/bird-listener/recordAnalyse.py
WorkingDirectory=/home/operator/bird-listener
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable bird-listener.service
sudo systemctl start bird-listener.service
```

## Architecture

### Core Components

**recordAnalyse.py** - Main application entry point
- Implements async producer-consumer pattern with two workers:
  - `recording_worker`: Continuously records 3-second audio chunks via ALSA
  - `analysis_worker`: Processes audio through BirdNET models
- On startup, initializes models, loads location-filtered species list, and creates SQLite database
- Detections are stored in `db/bird-observations.db` and optionally sent to weather-server API

**config.py** - Configuration management
- `BirdNetConfig` dataclass: Immutable configuration loaded from environment variables
- `from_env()`: Creates config from environment with sensible defaults
- `with_runtime_data()`: Adds labels and species list after initialization

**predict.py** - TFLite audio classification
- `TFLitePredictor`: Wraps BirdNET classification model inference
- Global singleton pattern via `init_predictor()` and `predict()` functions
- `flat_sigmoid()`: Applies sigmoid activation for probability calibration

**getSpecies.py** - Geographic/temporal species filtering
- `SpeciesPredictor`: Uses metadata model to filter species by location and week
- Reduces false positives by limiting predictions to regionally/seasonally appropriate species
- Global singleton pattern similar to predict.py

### Data Flow

1. Audio capture: ALSA → 3-second chunks at 48kHz
2. Queue management: Recording worker pushes to async queue
3. Analysis: Consumer pulls from queue, converts to [-1, 1] range
4. Species filtering: Geographic model pre-filters species list
5. Classification: Main model predicts species from audio
6. Post-processing: Apply sigmoid if configured, filter by confidence threshold
7. Storage: Insert detections to SQLite, optionally POST to weather-server

### Key Configuration Parameters

- `sample_rate`: 48000 Hz (BirdNET requirement)
- `min_confidence`: 0.05 (threshold for detections)
- `sigmoid_sensitivity`: 1.0 (calibration parameter)
- `location_filter_threshold`: 0.03 (species filtering threshold)
- `tflite_threads`: 1 (CPU threads for inference)

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
