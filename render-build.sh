#!/bin/bash

# Exit on error
set -e

echo "Updating package lists..."
apt-get update -y

echo "Installing dependencies..."
apt-get install -y wget unzip curl

echo "Installing Google Chrome..."
wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
echo "deb http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list
apt-get update -y
apt-get install -y google-chrome-stable

echo "Installing Chromedriver..."
CHROME_VERSION=$(google-chrome --version | awk '{print $3}' | cut -d '.' -f1)
CHROMEDRIVER_VERSION=$(curl -s "https://chromedriver.storage.googleapis.com/LATEST_RELEASE_$CHROME_VERSION")
wget -N https://chromedriver.storage.googleapis.com/$CHROMEDRIVER_VERSION/chromedriver_linux64.zip -P /tmp/
unzip -o /tmp/chromedriver_linux64.zip -d /tmp/
mv -f /tmp/chromedriver /usr/local/bin/chromedriver
chmod +x /usr/local/bin/chromedriver

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete!"
