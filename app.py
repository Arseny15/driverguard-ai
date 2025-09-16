#!/usr/bin/env python3
"""
DriveGuard AI Web Application

A modern web interface for the DriveGuard AI system with real-time video streaming,
alerts display, and configuration management.

Powered by ARS Technologies
"""

import os
import cv2
import base64
import threading
import time
import json
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
from flask_socketio import SocketIO, emit
import logging
from datetime import datetime

# Import our DriveGuard AI system
from driver_guard import DriveGuardAI
from config import Config
from security_auth import security_manager, require_auth, get_auth_routes

# Configure logging and create config instance
config = Config()
config.setup_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ars-driverguard-secure-key-2025'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
socketio = SocketIO(app, cors_allowed_origins="*")

class WebDriveGuard:
    """Web interface wrapper for DriveGuard AI."""
    
    def __init__(self):
        self.driver_guard = None
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
        # Load configuration
        self.config = config.get_detection_config()
    
    def initialize(self):
        """Initialize the DriveGuard AI system."""
        try:
            models_config = config.get_models_config()
            self.driver_guard = DriveGuardAI(
                config_path=models_config['face_detector']['config_path'],
                use_onnx=self.config.get('use_onnx', False),
                mode='gpu' if self.config.get('gpu_mode', False) else 'cpu'
            )
            self.stats['session_start'] = datetime.now()
            logger.info("DriveGuard AI initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize DriveGuard AI: {e}")
            return False
    
    def start_detection(self):
        """Start the detection process."""
        if not self.driver_guard:
            if not self.initialize():
                return False
        
        if self.is_running:
            return True
        
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                logger.error("Cannot open webcam")
                return False
            
            self.is_running = True
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
                
                # Process frame
                processed_frame, ear, mar, head_angle = self.driver_guard.process_frame(frame)
                self.current_frame = processed_frame
                self.stats['total_frames'] += 1
                
                # Check for alerts
                self._check_alerts(ear, mar, head_angle)
                
                # Emit frame to web clients
                self._emit_frame(processed_frame)
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                break
    
    def _check_alerts(self, ear, mar, head_angle):
        """Check for various alert conditions."""
        current_time = datetime.now()
        
        # Drowsiness detection
        if ear < self.config['eye_ar_threshold']:
            self.stats['drowsiness_detected'] += 1
            self._add_alert("DROWSINESS", "Eyes closed detected", current_time)
        
        # Attention loss detection
        if not (self.config['head_angle_min'] < head_angle < self.config['head_angle_max']):
            self.stats['attention_lost'] += 1
            self._add_alert("ATTENTION", "Look ahead!", current_time)
        
        # Yawning detection
        if mar > self.config['mouth_ar_threshold']:
            self._add_alert("YAWNING", "Yawning detected", current_time)
    
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
        if self.driver_guard:
            # Update thresholds in the driver guard instance
            self.driver_guard.thresholds.update({
                'eye_ar': self.config['eye_ar_threshold'],
                'mouth_ar': self.config['mouth_ar_threshold'],
                'head_angle_min': self.config['head_angle_min'],
                'head_angle_max': self.config['head_angle_max'],
                'consecutive_frames': self.config['consecutive_frames']
            })
            
            # Update detection mode
            if 'detection_mode' in new_config:
                self.driver_guard.detection_mode = new_config['detection_mode']
                logger.info(f"Detection mode changed to: {new_config['detection_mode']}")
                
                # Emit status update to clients
                socketio.emit('status', {
                    'message': f'Detection mode changed to {new_config["detection_mode"].upper()}',
                    'type': 'info'
                })
        logger.info(f"Configuration updated: {new_config}")

# Initialize web drive guard
web_drive_guard = WebDriveGuard()

# Add authentication routes
get_auth_routes(app, security_manager)

@app.route('/')
@require_auth
def index():
    """Main dashboard page."""
    return render_template('index.html')

@app.route('/api/start', methods=['POST'])
@require_auth
def start_detection():
    """Start detection API endpoint."""
    success = web_drive_guard.start_detection()
    return jsonify({'success': success})

@app.route('/api/stop', methods=['POST'])
@require_auth
def stop_detection():
    """Stop detection API endpoint."""
    web_drive_guard.stop_detection()
    return jsonify({'success': True})

@app.route('/api/stats')
@require_auth
def get_stats():
    """Get statistics API endpoint."""
    return jsonify(web_drive_guard.get_stats())

@app.route('/api/alerts')
@require_auth
def get_alerts():
    """Get alerts API endpoint."""
    limit = request.args.get('limit', 20, type=int)
    return jsonify(web_drive_guard.get_alerts(limit))

@app.route('/api/config', methods=['GET', 'POST'])
@require_auth
def config_api():
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
    
    # Run the application
    socketio.run(app, debug=True, host='0.0.0.0', port=3000)
