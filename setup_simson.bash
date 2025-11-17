#!/usr/bin/env bash
set -e

# Base working directory (the parent folder)
WORKDIR=$(pwd)

# Update and install basic dependencies
sudo apt update
sudo apt install -y python3-venv python3-pip

# Clone the SimSon repository if not already present
if [ ! -d "$WORKDIR/SimSon" ]; then
  git clone https://github.com/lee00206/SimSon.git
fi
cd SimSon

# Go back to parent to create venv outside SimSon
cd "$WORKDIR"

# Create and activate Python virtual environment in parent directory
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required Python dependencies (per SimSon README)
pip install numpy==1.22.4 \
            requests==2.32.3 \
            scikit-learn==1.5.0 \
            scipy==1.13.1 \
            tokenizers==0.19.1 \
            torch==2.3.1 \
            torchaudio==2.3.1 \
            torchvision==0.18.1 \
            tqdm==4.66.4 \
            transformers==4.42.3 \
            x-transformers==1.31.6

echo "✅ Environment setup complete."
echo ">>> Virtual environment: $WORKDIR/venv"
echo ">>> Repository: $WORKDIR/SimSon"
echo "⚠️  Remember to manually download pretrained model to:"
echo "⚠️  SimSon/models/pretrained/pretrained_best_model.pth"