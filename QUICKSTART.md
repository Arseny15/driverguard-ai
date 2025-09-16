# DriveGuard AI - Quick Start Guide

## 🚀 Quick Installation

1. **Clone and setup:**
```bash
git clone https://github.com/arstechnologies/driverguard-ai.git
cd driverguard-ai
chmod +x install.sh
./install.sh
```

2. **Activate environment:**
```bash
source venv/bin/activate
```

3. **Start the web interface:**
```bash
python app.py
```

4. **Open your browser:**
Navigate to `http://localhost:5000`

**Login Credentials:**
- Passcode: `ARSGuard`

## 🎯 Features

- **Real-time driver monitoring** with webcam feed
- **Drowsiness detection** using eye aspect ratio analysis
- **Attention tracking** with head pose estimation
- **Yawning detection** through mouth analysis
- **Live statistics** and alert history
- **Configurable thresholds** and settings
- **Modern web interface** with responsive design

## 🛠️ Usage

### Web Interface (Recommended)
1. Click "Start" to begin monitoring
2. Adjust detection thresholds in the configuration panel
3. View real-time alerts and statistics
4. Click "Stop" to end monitoring

### Command Line Interface
```bash
# Basic detection
python driver_guard.py --mode basic

# Advanced 3D analysis
python driver_guard.py --mode advanced --3d

# ONNX optimized
python driver_guard.py --mode onnx
```

## ⚙️ Configuration

Edit `configs/driver_guard.yml` to customize:
- Detection thresholds
- Audio settings
- Web interface options
- Model configurations

## 📁 Project Structure

```
DriveGuard-AI/
├── app.py                 # Web application
├── driver_guard.py        # Main detection system
├── config.py             # Configuration management
├── requirements.txt      # Dependencies
├── install.sh           # Installation script
├── templates/           # Web templates
├── static/             # Static web assets (includes ARS logo)
├── configs/            # Configuration files
├── audio/              # Audio alert files
├── logs/               # Log files
└── models/             # Model files
```

## 🔧 Troubleshooting

### Common Issues

1. **Webcam not detected:**
   - Check camera permissions
   - Ensure no other applications are using the camera

2. **Audio not working:**
   - Check audio file paths in configuration
   - Ensure audio files exist in `audio/` directory

3. **Model files missing:**
   - Download required model files as shown in installation
   - Check file paths in configuration

4. **Performance issues:**
   - Try ONNX mode for better performance
   - Adjust detection thresholds
   - Use GPU mode if available

### Getting Help

- Check the logs in `logs/driver_guard.log`
- Review configuration in `configs/driver_guard.yml`
- Ensure all dependencies are installed correctly

## 🎉 You're Ready!

DriveGuard AI is now ready to help monitor driver behavior and enhance road safety!

**Powered by ARS Technologies** - Advanced driver monitoring solutions for a safer tomorrow.
