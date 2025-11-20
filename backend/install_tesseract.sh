#!/bin/bash
set -e

echo "📦 Installing latest AI-based Tesseract OCR..."

apt-get update -y
apt-get install -y wget gnupg lsb-release apt-utils

echo "deb http://deb.debian.org/debian bookworm-backports main contrib non-free" >> /etc/apt/sources.list
apt-get update -y

apt-get install -y -t bookworm-backports \
  tesseract-ocr \
  tesseract-ocr-eng \
  tesseract-ocr-hin \
  tesseract-ocr-kan

echo "✅ Installed Tesseract version:"
tesseract --version || (echo "❌ Install failed!" && exit 1)
which tesseract

export TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
export PATH=$PATH:/usr/bin
