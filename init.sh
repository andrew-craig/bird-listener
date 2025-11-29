#!/bin/bash
set -e  # Exit on error

echo "Bird Listener Initialization Script"
echo "===================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Create birdnet_analyzer directory if it doesn't exist
echo "Creating birdnet_analyzer directory..."
mkdir -p birdnet_analyzer

# Download BirdNET-Analyzer module files
echo "Downloading BirdNET-Analyzer module from GitHub..."
REPO_URL="https://github.com/birdnet-team/BirdNET-Analyzer"
BRANCH="main"
MODULE_PATH="birdnet_analyzer"

# Download individual files from the birdnet_analyzer directory
# Using GitHub's raw content URL
RAW_BASE_URL="https://raw.githubusercontent.com/birdnet-team/BirdNET-Analyzer/${BRANCH}/${MODULE_PATH}"

echo "Downloading Python module files..."
# Create a temporary directory to clone just what we need
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Clone the repository with sparse checkout
git clone --depth 1 --filter=blob:none --sparse "$REPO_URL" .
git sparse-checkout set "$MODULE_PATH"

# Copy the birdnet_analyzer directory to the project
cd -
rm -rf birdnet_analyzer
cp -r "$TEMP_DIR/$MODULE_PATH" ./

# Clean up
rm -rf "$TEMP_DIR"

echo "BirdNET-Analyzer module downloaded successfully"

# Create checkpoints directory for models
echo "Creating checkpoints directory..."
mkdir -p birdnet_analyzer/checkpoints

# Download the V2.4 model
echo "Downloading V2.4 model (this may take a while)..."
cd birdnet_analyzer/checkpoints

MODEL_URL="https://drive.google.com/file/d/1ixYBPbZK2Fh1niUQzadE2IWTFZlwATa3"
if command -v wget &> /dev/null; then
    wget -O V2.4.zip "$MODEL_URL"
elif command -v curl &> /dev/null; then
    curl -L -o V2.4.zip "$MODEL_URL"
else
    echo "Error: Neither wget nor curl is installed"
    exit 1
fi

# Unzip the model
echo "Extracting model files..."
if command -v unzip &> /dev/null; then
    unzip -q V2.4.zip
    rm V2.4.zip
else
    echo "Warning: unzip is not installed. Model file V2.4.zip needs to be extracted manually"
fi

cd ../..

# Copy config template if config doesn't exist
if [ ! -f config.yaml ]; then
    if [ -f config.template.yaml ]; then
        echo "Creating config.yaml from template..."
        cp config.template.yaml config.yaml
        echo "Please edit config.yaml and add your latitude and longitude"
    else
        echo "Warning: config.template.yaml not found"
    fi
fi

echo ""
echo "Initialization complete!"
echo ""
echo "Next steps:"
echo "1. Edit config.yaml and add your latitude and longitude"
echo "2. Install ffmpeg if not already installed:"
echo "   - macOS: brew install ffmpeg"
echo "   - Linux: sudo apt-get install ffmpeg"
echo "3. Create and activate a virtual environment:"
echo "   python3 -m venv venv"
echo "   source venv/bin/activate  # On Windows: venv\\Scripts\\activate"
echo "4. Install dependencies:"
echo "   pip install -e ."
echo "5. Run the bird listener application"
