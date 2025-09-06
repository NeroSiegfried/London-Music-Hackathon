#!/bin/bash

# ABRSM AI Music Feedback System Setup Script

echo "🎼 Setting up ABRSM AI Music Feedback System..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install dependencies
echo "⚙️  Installing dependencies..."
source venv/bin/activate
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "To use the system:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Set your Google API key: export GOOGLE_API_KEY='your_key_here'"
echo "3. Try the demo: python enhanced_main.py --demo"
echo ""
echo "🎵 Happy music analyzing!"
