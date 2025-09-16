# DriveGuard AI

## Project Description
DriveGuard AI is an intelligent computer vision system designed to enhance road safety through real-time driver behavior monitoring. This project leverages advanced deep learning models to detect and alert drivers about potentially dangerous behaviors, including phone usage while driving, drowsiness, and loss of attention. The primary goal is to help reduce road accidents by providing immediate feedback to drivers about their potentially hazardous actions.

**Powered by ARS Technologies**

## Features
- **Phone Use Detection**: Advanced YOLOv5 model integration to identify drivers using their phones during driving.
- **Drowsiness Detection**: Sophisticated facial landmark analysis to monitor signs of driver fatigue and drowsiness.
- **Head Pose Analysis**: Real-time head orientation tracking to detect when drivers are not looking at the road.
- **Real-Time Alerts**: Intelligent alert system with visual warnings when risky behaviors are detected.
- **Modern Web Interface**: Clean, responsive web-based dashboard for monitoring and configuration.
- **Security Features**: Login system with session management for secure access.

## Requirements

- Python >= 3.8
- OpenCV >= 4.5.0
- Flask (for web interface)
- NumPy, SciPy

## 🚀 Installation and Configuration

1. **Clone the repository:**
```bash
git clone https://github.com/arstechnologies/driverguard-ai.git
cd driverguard-ai
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Start the application:**
   - The system will work in demo mode with basic face detection

## 🤖 How to Use

### Web Interface (Recommended)
```bash
python app.py
```
Then open your browser to `http://localhost:5000` for the modern web interface.

### Command Line Interface
1. **Basic Detection:**
```bash
python driver_guard.py --mode basic
```

2. **Advanced 3D analysis:**
```bash
python driver_guard.py --mode advanced --3d
```

3. **ONNX optimized:**
```bash
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