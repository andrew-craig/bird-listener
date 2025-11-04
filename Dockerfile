# Use uv-enabled Python 3.11 base image
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    alsa-utils \
    libasound2 \
    libasound2-dev \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone BirdNET-Analyzer repository
#RUN git clone https://github.com/kahst/BirdNET-Analyzer.git && \
#    cp -r BirdNET-Analyzer/birdnet_analyzer /app/birdnet_analyzer && \
#    rm -rf BirdNET-Analyzer

# Copy project files
COPY pyproject.toml .
COPY recordAnalyse.py .
COPY birdnet_analyzer .

RUN uv sync

# Copy the record-analyse script into BirdNET-Analyzer directory


# Download and setup the models
#RUN mkdir -p birdnet_analyzer/checkpoints
#WORKDIR /app/birdnet_analyzer/checkpoints
#RUN wget https://tuc.cloud/index.php/s/886x39f5N3sdsAM/download/V2.4.zip
#RUN unzip V2.4.zip
#RUN rm V2.4.zip
#WORKDIR /app


# Create directories for recordings and database
RUN mkdir -p /app/recordings /app/db

# Volume for persistent data
VOLUME ["/app/recordings", "/app/db"]


# Run the application
CMD ["uv", "run", "recordAnalyse.py"]
