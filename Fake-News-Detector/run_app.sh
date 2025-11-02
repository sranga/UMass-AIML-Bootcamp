#!/bin/zsh

# Python 3.12 didn't work for me. So, installed 3.11
conda create -n fakenews python=3.11
conda activate fakenews

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt --upgrade

# Test the API
uvicorn api:app --reload --port 8000

