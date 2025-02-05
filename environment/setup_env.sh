#!/bin/bash

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK stopwords
python -m nltk.downloader stopwords

echo 'Environment setup completed. Activate it using source venv/bin/activate.'
