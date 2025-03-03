#!/bin/bash

# Exit on error
set -e

echo "Downloading Chrome..."
wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb -O /tmp/chrome.deb

echo "Downloading Chromedriver..."
CHROMEDRIVER_VERSION=$(curl -s "https://chromedriver.storage.googleapis.com/LATEST_RELEASE")
wget -q https://chromedriver.storage.googleapis.com/$CHROMEDRIVER_VERSION/chromedriver_linux64.zip -O /tmp/chromedriver.zip
unzip -q /tmp/chromedriver.zip -d /tmp/

echo "Setting up permissions..."
chmod +x /tmp/chrome.deb
chmod +x /tmp/chromedriver

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete!"
