#!/bin/bash
echo "=== PRAHARI Setup ==="
cd /Users/mohitmudgil/Desktop/prahari/PRAHARI

echo "1. Creating Python Virtual Environment (Python 3.11 for Moirai)..."
python3.11 -m venv venv
source venv/bin/activate

echo "2. Installing Core Dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install pandas numpy scipy scikit-learn matplotlib
pip install xarray netCDF4 cdflib PyWavelets
pip install transformers peft huggingface_hub
pip install fastapi uvicorn
pip install jupyterlab

echo "3. Installing Moirai (Salesforce uni2ts)..."
pip install uni2ts

echo "=== PRAHARI Setup Complete ==="

