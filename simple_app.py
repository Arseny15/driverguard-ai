#!/usr/bin/env python3
"""
DriveGuard AI - Simplified Web Application

A simplified version that works without dlib for demonstration purposes.

Powered by ARS Technologies
"""

import os
import cv2
import base64
import threading
import time
import json
import logging
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

class SimpleDriveGuard:
    """Simplified web interface for DriveGuard AI."""
    
    def __init__(self):
        self.is_running = False
        self.camera = None
        self.current_frame = None
        self.alerts = []
        self.stats = {
            'total_frames': 0,
            'drowsiness_detected': 0,
            'attention_lost': 0,
            'phone_usage': 0,
            'session_start': None
        }
        self.config = {
            'eye_ar_threshold': 0.33,
            'mouth_ar_threshold': 0.7,
            'head_angle_min': 75,
            'head_angle_max': 110,
            'consecutive_frames': 6,
            'detection_mode': 'basic',
            'use_onnx': False
        }
    
    def start_detection(self):
        """Start the detection process."""
        if self.is_running:
            return True
        
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                logger.error("Cannot open webcam")
                return False
            
            self.is_running = True
            self.stats['session_start'] = datetime.now()
            self.detection_thread = threading.Thread(target=self._detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
            logger.info("Detection started")
            return True
        except Exception as e:
            logger.error(f"Failed to start detection: {e}")
            return False
    
    def stop_detection(self):
        """Stop the detection process."""
        self.is_running = False
        if self.camera:
            self.camera.release()
        logger.info("Detection stopped")
    
    def _detection_loop(self):
        """Main detection loop running in separate thread."""
        while self.is_running:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                # Simple face detection using OpenCV's built-in cascade classifier
                processed_frame = self._process_frame_simple(frame)
                self.current_frame = processed_frame
                self.stats['total_frames'] += 1
                
                # Simulate some detection for demo purposes
                self._simulate_detections()
                
                # Emit frame to web clients
                self._emit_frame(processed_frame)
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                break
    
    def _process_frame_simple(self, frame):
        """Simple frame processing without dlib."""
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load OpenCV's face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Add some demo information
        cv2.putText(frame, f'Frames: {self.stats["total_frames"]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Drowsiness: {self.stats["drowsiness_detected"]}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Attention Lost: {self.stats["attention_lost"]}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def _simulate_detections(self):
        """Simulate some detections for demo purposes."""
        # Simulate occasional drowsiness detection
        if self.stats['total_frames'] % 100 == 0:
            self.stats['drowsiness_detected'] += 1
            self._add_alert("DROWSINESS", "Simulated drowsiness detected", datetime.now())
        
        # Simulate occasional attention loss
        if self.stats['total_frames'] % 150 == 0:
            self.stats['attention_lost'] += 1
            self._add_alert("ATTENTION", "Simulated attention loss", datetime.now())
    
    def _add_alert(self, alert_type, message, timestamp):
        """Add a new alert to the system."""
        alert = {
            'id': len(self.alerts) + 1,
            'type': alert_type,
            'message': message,
            'timestamp': timestamp.strftime('%H:%M:%S'),
            'severity': 'HIGH' if alert_type in ['DROWSINESS', 'ATTENTION'] else 'MEDIUM'
        }
        self.alerts.append(alert)
        
        # Keep only last 50 alerts
        if len(self.alerts) > 50:
            self.alerts = self.alerts[-50:]
        
        # Emit alert to web clients
        socketio.emit('new_alert', alert)
    
    def _emit_frame(self, frame):
        """Emit frame data to web clients."""
        try:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_data = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('frame_data', {'frame': frame_data})
        except Exception as e:
            logger.error(f"Error emitting frame: {e}")
    
    def get_stats(self):
        """Get current statistics."""
        return self.stats
    
    def get_alerts(self, limit=20):
        """Get recent alerts."""
        return self.alerts[-limit:] if self.alerts else []
    
    def update_config(self, new_config):
        """Update configuration."""
        self.config.update(new_config)
        logger.info(f"Configuration updated: {new_config}")

# Initialize web drive guard
web_drive_guard = SimpleDriveGuard()

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')

@app.route('/api/start', methods=['POST'])
def start_detection():
    """Start detection API endpoint."""
    success = web_drive_guard.start_detection()
    return jsonify({'success': success})

@app.route('/api/stop', methods=['POST'])
def stop_detection():
    """Stop detection API endpoint."""
    web_drive_guard.stop_detection()
    return jsonify({'success': True})

@app.route('/api/stats')
def get_stats():
    """Get statistics API endpoint."""
    return jsonify(web_drive_guard.get_stats())

@app.route('/api/alerts')
def get_alerts():
    """Get alerts API endpoint."""
    limit = request.args.get('limit', 20, type=int)
    return jsonify(web_drive_guard.get_alerts(limit))

@app.route('/api/config', methods=['GET', 'POST'])
def config():
    """Configuration API endpoint."""
    if request.method == 'GET':
        return jsonify(web_drive_guard.config)
    elif request.method == 'POST':
        new_config = request.get_json()
        web_drive_guard.update_config(new_config)
        return jsonify({'success': True})

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info('Client connected')
    emit('status', {'message': 'Connected to DriveGuard AI'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info('Client disconnected')

@socketio.on('request_frame')
def handle_frame_request():
    """Handle frame request from client."""
    if web_drive_guard.current_frame is not None:
        web_drive_guard._emit_frame(web_drive_guard.current_frame)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    print("🚀 Starting DriveGuard AI...")
    print("📱 Open your browser to: http://localhost:8080")
    print("⚠️  Note: This is a simplified version without advanced face detection")
    print("🛑 Press Ctrl+C to stop")
    
    # Run the application
    socketio.run(app, debug=True, host='0.0.0.0', port=8080)
