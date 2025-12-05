#!/bin/bash
set -e  # Exit on error

echo "Bird Listener Initialization Script"
echo "===================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Create models directory
echo "Creating models directory..."
mkdir -p models

# Download the V2.4 model
echo "Downloading V2.4 model (this may take a while)..."
cd models

MODEL_URL="https://drive.usercontent.google.com/download?id=1ixYBPbZK2Fh1niUQzadE2IWTFZlwATa3&export=download&authuser=0&confirm=t&uuid=b3c4c3ea-42eb-458a-baeb-104f03ef93b4"
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
