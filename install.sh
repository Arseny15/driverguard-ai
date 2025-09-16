#!/bin/bash

# DriverGuard AI Installation Script

echo "🚀 Installing DriverGuard AI..."

# Check if Python 3.8+ is installed
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8 or higher is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p models
mkdir -p audio
mkdir -p static/css
mkdir -p static/js

# Download required model files (if not present)
if [ ! -f "shape_predictor_81_face_landmarks (1).dat" ]; then
    echo "⚠️  Please download the dlib shape predictor file:"
    echo "   wget http://dlib.net/files/shape_predictor_81_face_landmarks.dat.bz2"
    echo "   bunzip2 shape_predictor_81_face_landmarks.dat.bz2"
    echo "   mv shape_predictor_81_face_landmarks.dat 'shape_predictor_81_face_landmarks (1).dat'"
fi

if [ ! -f "models/yolov5m.pt" ]; then
    echo "⚠️  Please download the YOLOv5 model:"
    echo "   wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5m.pt -O models/yolov5m.pt"
fi

echo "✅ Installation completed!"
echo ""
echo "🎯 To start the web interface:"
echo "   source venv/bin/activate"
echo "   python app.py"
echo ""
echo "🎯 To start the command line interface:"
echo "   source venv/bin/activate"
echo "   python driver_guard.py --help"
echo ""
echo "🌐 Web interface will be available at: http://localhost:5000"
