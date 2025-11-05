#!/bin/zsh

# If Python 3.12 doesn't work uncomment the following lines
#conda init
#conda create -n fakenews python=3.11
#conda activate fakenews

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt --upgrade

# Test the API
uvicorn api:app --reload --port 8000

