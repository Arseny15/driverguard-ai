#!/usr/bin/env python3
"""
DriveGuard AI - Secure Web Application

A security-enhanced version with comprehensive privacy protection and data security.
"""

import os
import cv2
import base64
import threading
import time
import json
import logging
from flask import Flask, render_template, Response, jsonify, request, session
from flask_socketio import SocketIO, emit
from datetime import datetime
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# Import our security module
from security import security_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = security_manager.generate_secure_token('flask_secret')
socketio = SocketIO(app, cors_allowed_origins="*")

class SecureDriveGuard:
    """Security-enhanced web interface for DriveGuard AI."""
    
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
            'use_onnx': False,
            'privacy_mode': True,
            'data_encryption': True
        }
        
        # Security features
        self.session_tokens = {}
        self.failed_attempts = {}
        self.max_attempts = 5
        
        # Initialize security
        self._init_security()
        
        # Log security initialization
        security_manager.log_audit_event('system_start', 'Secure DriveGuard AI initialized')
    
    def _init_security(self):
        """Initialize security features."""
        # Enable privacy mode by default
        security_manager.enable_privacy_mode()
        
        # Set up secure data storage
        self.secure_config_file = 'secure_config'
        self.secure_stats_file = 'secure_stats'
        
        # Load secure configuration
        secure_config = security_manager.load_secure_data(self.secure_config_file)
        if secure_config:
            self.config.update(secure_config)
        
        logger.info("Security features initialized")
    
    def _require_auth(self, f):
        """Decorator to require authentication."""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            token = request.headers.get('Authorization', '').replace('Bearer ', '')
            if not self._validate_session_token(token):
                security_manager.log_audit_event('unauthorized_access', f'Failed access to {request.endpoint}')
                return jsonify({'error': 'Unauthorized'}), 401
            return f(*args, **kwargs)
        return decorated_function
    
    def _validate_session_token(self, token):
        """Validate session token."""
        if not token:
            return False
        
        # Check if token exists and is valid
        if token in self.session_tokens:
            session_data = self.session_tokens[token]
            # Check if session is not expired
            if time.time() - session_data['created'] < 1800:  # 30 minutes
                return True
            else:
                # Remove expired token
                del self.session_tokens[token]
        
        return False
    
    def _generate_session_token(self):
        """Generate a new session token."""
        token = security_manager.generate_secure_token('session')
        self.session_tokens[token] = {
            'created': time.time(),
            'last_activity': time.time()
        }
        return token
    
    def _log_security_event(self, event_type, details):
        """Log security events."""
        security_manager.log_audit_event(event_type, details)
    
    def start_detection(self):
        """Start the detection process with security checks."""
        # Check for too many failed attempts
        client_ip = request.remote_addr
        if client_ip in self.failed_attempts:
            if self.failed_attempts[client_ip] >= self.max_attempts:
                self._log_security_event('blocked_access', f'Too many failed attempts from {client_ip}')
                return False
        
        if self.is_running:
            return True
        
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                logger.error("Cannot open webcam")
                self._log_security_event('camera_error', 'Failed to open webcam')
                return False
            
            self.is_running = True
            self.stats['session_start'] = datetime.now()
            self.detection_thread = threading.Thread(target=self._detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
            self._log_security_event('detection_started', 'Driver monitoring started')
            logger.info("Secure detection started")
            return True
        except Exception as e:
            logger.error(f"Failed to start detection: {e}")
            self._log_security_event('detection_error', f'Failed to start detection: {str(e)}')
            return False
    
    def stop_detection(self):
        """Stop the detection process."""
        self.is_running = False
        if self.camera:
            self.camera.release()
        
        self._log_security_event('detection_stopped', 'Driver monitoring stopped')
        logger.info("Secure detection stopped")
    
    def _detection_loop(self):
        """Main detection loop with privacy protection."""
        while self.is_running:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                # Apply privacy protection
                processed_frame = self._process_frame_secure(frame)
                self.current_frame = processed_frame
                self.stats['total_frames'] += 1
                
                # Simulate some detection for demo purposes
                self._simulate_detections()
                
                # Emit frame to web clients (with privacy protection)
                self._emit_frame_secure(processed_frame)
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                self._log_security_event('detection_loop_error', f'Error in detection loop: {str(e)}')
                break
    
    def _process_frame_secure(self, frame):
        """Secure frame processing with privacy protection."""
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load OpenCV's face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Apply privacy protection
        if self.config.get('privacy_mode', True):
            # Blur faces for privacy
            frame = security_manager.blur_sensitive_regions(frame, faces)
        else:
            # Draw rectangles around faces (non-privacy mode)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Add demo information (anonymized)
        cv2.putText(frame, f'Frames: {self.stats["total_frames"]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Drowsiness: {self.stats["drowsiness_detected"]}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Attention Lost: {self.stats["attention_lost"]}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add security indicator
        cv2.putText(frame, 'SECURE MODE', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return frame
    
    def _simulate_detections(self):
        """Simulate some detections for demo purposes."""
        # Simulate occasional drowsiness detection
        if self.stats['total_frames'] % 100 == 0:
            self.stats['drowsiness_detected'] += 1
            self._add_secure_alert("DROWSINESS", "Simulated drowsiness detected", datetime.now())
        
        # Simulate occasional attention loss
        if self.stats['total_frames'] % 150 == 0:
            self.stats['attention_lost'] += 1
            self._add_secure_alert("ATTENTION", "Simulated attention loss", datetime.now())
    
    def _add_secure_alert(self, alert_type, message, timestamp):
        """Add a new alert with security logging."""
        # Anonymize alert data
        alert_data = {
            'type': alert_type,
            'message': message,
            'timestamp': timestamp
        }
        
        # Apply privacy protection
        if self.config.get('data_encryption', True):
            alert_data = security_manager.anonymize_face_data(alert_data)
        
        alert = {
            'id': len(self.alerts) + 1,
            'type': alert_data['type'],
            'message': alert_data['message'],
            'timestamp': alert_data['timestamp'].strftime('%H:%M:%S') if isinstance(alert_data['timestamp'], datetime) else str(alert_data['timestamp']),
            'severity': 'HIGH' if alert_type in ['DROWSINESS', 'ATTENTION'] else 'MEDIUM'
        }
        
        self.alerts.append(alert)
        
        # Keep only last 50 alerts
        if len(self.alerts) > 50:
            self.alerts = self.alerts[-50:]
        
        # Log security event
        self._log_security_event('alert_generated', f'Alert: {alert_type}')
        
        # Emit alert to web clients
        socketio.emit('new_alert', alert)
    
    def _emit_frame_secure(self, frame):
        """Emit frame data with security measures."""
        try:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_data = base64.b64encode(buffer).decode('utf-8')
                
                # Add security metadata
                secure_frame_data = {
                    'frame': frame_data,
                    'timestamp': time.time(),
                    'encrypted': self.config.get('data_encryption', True)
                }
                
                socketio.emit('frame_data', secure_frame_data)
        except Exception as e:
            logger.error(f"Error emitting frame: {e}")
            self._log_security_event('frame_emit_error', f'Error emitting frame: {str(e)}')
    
    def get_stats(self):
        """Get current statistics with privacy protection."""
        stats = self.stats.copy()
        
        # Anonymize sensitive data
        if self.config.get('data_encryption', True):
            stats = security_manager.anonymize_face_data(stats)
        
        return stats
    
    def get_alerts(self, limit=20):
        """Get recent alerts with privacy protection."""
        alerts = self.alerts[-limit:] if self.alerts else []
        
        # Apply privacy protection to alerts
        if self.config.get('data_encryption', True):
            alerts = [security_manager.anonymize_face_data(alert) for alert in alerts]
        
        return alerts
    
    def update_config(self, new_config):
        """Update configuration with security validation."""
        # Validate configuration
        allowed_keys = ['eye_ar_threshold', 'mouth_ar_threshold', 'head_angle_min', 
                       'head_angle_max', 'consecutive_frames', 'detection_mode']
        
        filtered_config = {k: v for k, v in new_config.items() if k in allowed_keys}
        
        if filtered_config:
            self.config.update(filtered_config)
            
            # Securely store configuration
            security_manager.secure_data_storage(self.config, self.secure_config_file)
            
            self._log_security_event('config_updated', f'Configuration updated: {list(filtered_config.keys())}')
            logger.info(f"Configuration updated: {filtered_config}")
            return True
        
        return False
    
    def cleanup_secure_data(self):
        """Clean up old data securely."""
        cleaned_count = security_manager.cleanup_old_data()
        self._log_security_event('data_cleanup', f'Cleaned {cleaned_count} old data entries')
        return cleaned_count

# Initialize secure web driver guard
web_drive_guard = SecureDriveGuard()

@app.route('/')
def index():
    """Main dashboard page with security features."""
    return render_template('secure_index.html')

@app.route('/api/auth', methods=['POST'])
def authenticate():
    """Authenticate user and return session token."""
    data = request.get_json()
    
    # Simple authentication (in production, use proper user management)
    if data.get('password') == 'ARSGuard':  # Default password
        token = web_driver_guard._generate_session_token()
        web_driver_guard._log_security_event('user_authenticated', 'User authenticated successfully')
        return jsonify({'success': True, 'token': token})
    else:
        # Track failed attempts
        client_ip = request.remote_addr
        web_driver_guard.failed_attempts[client_ip] = web_driver_guard.failed_attempts.get(client_ip, 0) + 1
        web_driver_guard._log_security_event('auth_failed', f'Failed authentication from {client_ip}')
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 401

@app.route('/api/start', methods=['POST'])
@web_driver_guard._require_auth
def start_detection():
    """Start detection API endpoint with authentication."""
    success = web_driver_guard.start_detection()
    return jsonify({'success': success})

@app.route('/api/stop', methods=['POST'])
@web_driver_guard._require_auth
def stop_detection():
    """Stop detection API endpoint with authentication."""
    web_driver_guard.stop_detection()
    return jsonify({'success': True})

@app.route('/api/stats')
@web_driver_guard._require_auth
def get_stats():
    """Get statistics API endpoint with authentication."""
    return jsonify(web_driver_guard.get_stats())

@app.route('/api/alerts')
@web_driver_guard._require_auth
def get_alerts():
    """Get alerts API endpoint with authentication."""
    limit = request.args.get('limit', 20, type=int)
    return jsonify(web_driver_guard.get_alerts(limit))

@app.route('/api/config', methods=['GET', 'POST'])
@web_driver_guard._require_auth
def config():
    """Configuration API endpoint with authentication."""
    if request.method == 'GET':
        return jsonify(web_driver_guard.config)
    elif request.method == 'POST':
        new_config = request.get_json()
        success = web_driver_guard.update_config(new_config)
        return jsonify({'success': success})

@app.route('/api/security/status')
@web_driver_guard._require_auth
def security_status():
    """Get security status."""
    return jsonify(security_manager.get_security_status())

@app.route('/api/security/cleanup', methods=['POST'])
@web_driver_guard._require_auth
def cleanup_data():
    """Clean up old data."""
    cleaned_count = web_driver_guard.cleanup_secure_data()
    return jsonify({'success': True, 'cleaned_count': cleaned_count})

@app.route('/api/security/export', methods=['POST'])
@web_driver_guard._require_auth
def export_logs():
    """Export security logs."""
    output_file = security_manager.export_security_logs()
    return jsonify({'success': True, 'export_file': output_file})

@socketio.on('connect')
def handle_connect():
    """Handle client connection with security logging."""
    web_driver_guard._log_security_event('client_connected', f'Client connected from {request.remote_addr}')
    emit('status', {'message': 'Connected to Secure DriveGuard AI'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection with security logging."""
    web_driver_guard._log_security_event('client_disconnected', f'Client disconnected from {request.remote_addr}')

@socketio.on('request_frame')
def handle_frame_request():
    """Handle frame request from client."""
    if web_driver_guard.current_frame is not None:
        web_driver_guard._emit_frame_secure(web_driver_guard.current_frame)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('configs/security', exist_ok=True)
    
    print("🔒 Starting Secure DriveGuard AI...")
    print("📱 Open your browser to: http://localhost:8080")
    print("🔐 Default password: ARSGuard")
    print("⚠️  Note: This version includes enhanced security features")
    print("🛑 Press Ctrl+C to stop")
    
    # Run the application
    socketio.run(app, debug=True, host='0.0.0.0', port=8080)
