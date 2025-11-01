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

# Copy project files
COPY config.template.yaml .
COPY recordAnalyse.py .
COPY pyproject.toml .

# Clone BirdNET-Analyzer repository
RUN git clone https://github.com/kahst/BirdNET-Analyzer.git

# Copy the record-analyse script into BirdNET-Analyzer directory
RUN cp recordAnalyse.py BirdNET-Analyzer/

# Download and setup the models
RUN mkdir -p BirdNET-Analyzer/birdnet_analyzer/checkpoints
WORKDIR /app/BirdNET-Analyzer/birdnet_analyzer/checkpoints
RUN wget https://tuc.cloud/index.php/s/886x39f5N3sdsAM/download/V2.4.zip
RUN unzip V2.4.zip
RUN rm V2.4.zip
WORKDIR /app

# Install Python dependencies using uv
RUN uv sync

# Create directories for recordings and database
RUN mkdir -p /app/recordings /app/db

# Set working directory to BirdNET-Analyzer
WORKDIR /app/BirdNET-Analyzer

# Volume for persistent data
VOLUME ["/app/recordings", "/app/db"]

# Run the application
CMD ["python", "recordAnalyse.py"]
