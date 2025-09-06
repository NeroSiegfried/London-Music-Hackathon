#!/bin/bash

# ABRSM AI Music Feedback System Setup Script

echo "🎼 Setting up ABRSM AI Music Feedback System..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "../venv" ]; then
    echo "📦 Creating virtual environment..."
    cd ..
    python3 -m venv venv
    cd scripts
fi

# Activate virtual environment and install dependencies
echo "⚙️  Installing dependencies..."
cd ..
source venv/bin/activate
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "To use the system:"
echo "1. Change to project directory: cd .."
echo "2. Activate the environment: source venv/bin/activate"
echo "3. Set your Google API key: export GOOGLE_API_KEY='your_key_here'"
echo "4. Try the demo: python enhanced_main_fixed.py --demo"
echo ""
echo "🎵 Happy music analyzing!"
