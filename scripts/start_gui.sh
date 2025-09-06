#!/bin/bash

# ABRSM GUI Startup Script
# This script activates the virtual environment and launches the GUI

echo "🎵 ABRSM AI Music Feedback System"
echo "Starting GUI interface..."
echo

# Change to project directory
cd "$(dirname "$0")/.."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if Python packages are installed
echo "🔍 Checking dependencies..."
python -c "import tkinter, matplotlib, numpy, librosa" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies!"
    echo "Installing required packages..."
    pip install -r requirements.txt
fi

# Launch GUI
echo "🚀 Launching GUI..."
python enhanced_gui_interface.py

echo "👋 GUI session ended"
