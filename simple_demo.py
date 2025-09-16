#!/usr/bin/env python3
"""
Simple demo version of DriveGuard AI
"""

from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from flask_socketio import SocketIO, emit
import logging

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ars-driverguard-secure-key-2025'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
socketio = SocketIO(app, cors_allowed_origins="*")

# Simple security manager
class SimpleSecurityManager:
    def __init__(self):
        self.default_passcode = "ARSGuard"
        self.active_sessions = {}
    
    def authenticate(self, password):
        if password == self.default_passcode:
            import secrets
            token = secrets.token_urlsafe(32)
            self.active_sessions[token] = {'created_at': time.time()}
            return True, token
        return False, "Invalid access code"
    
    def validate_session(self, token):
        return token in self.active_sessions

security_manager = SimpleSecurityManager()

def require_auth(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = session.get('auth_token')
        if not token or not security_manager.validate_session(token):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/api/auth', methods=['POST'])
def authenticate():
    try:
        data = request.get_json()
        password = data.get('password', '')
        
        success, result = security_manager.authenticate(password)
        
        if success:
            session['auth_token'] = result
            return jsonify({
                'success': True,
                'token': result,
                'message': 'Authentication successful'
            })
        else:
            return jsonify({
                'success': False,
                'message': result
            }), 401
    except Exception as e:
        return jsonify({
            'success': False,
            'message': 'Authentication failed'
        }), 500

@app.route('/api/logout', methods=['POST'])
@require_auth
def logout():
    token = session.get('auth_token')
    if token in security_manager.active_sessions:
        del security_manager.active_sessions[token]
    session.pop('auth_token', None)
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/auth/status')
def auth_status():
    token = session.get('auth_token')
    if token and security_manager.validate_session(token):
        return jsonify({'authenticated': True})
    else:
        return jsonify({'authenticated': False})

@app.route('/')
@require_auth
def index():
    return render_template('index.html')

@app.route('/api/start', methods=['POST'])
@require_auth
def start_detection():
    return jsonify({'success': True, 'message': 'Demo mode - detection started'})

@app.route('/api/stop', methods=['POST'])
@require_auth
def stop_detection():
    return jsonify({'success': True, 'message': 'Demo mode - detection stopped'})

@app.route('/api/stats')
@require_auth
def get_stats():
    return jsonify({
        'total_frames': 100,
        'drowsiness_detected': 5,
        'attention_lost': 3,
        'phone_usage': 0,
        'session_start': '2024-01-01T00:00:00'
    })

@app.route('/api/alerts')
@require_auth
def get_alerts():
    return jsonify([])

@app.route('/api/config', methods=['GET', 'POST'])
@require_auth
def config():
    if request.method == 'GET':
        return jsonify({
            'eye_ar_threshold': 0.33,
            'mouth_ar_threshold': 0.7,
            'detection_mode': '2d_sparse'
        })
    elif request.method == 'POST':
        return jsonify({'success': True})

@socketio.on('connect')
def handle_connect():
    emit('status', {'message': 'Connected to DriveGuard AI'})

@socketio.on('disconnect')
def handle_disconnect():
    pass

@socketio.on('request_frame')
def handle_frame_request():
    pass

if __name__ == '__main__':
    import time
    import os
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("🚀 Starting DriveGuard AI Demo Server...")
    print("🔐 Login with passcode: ARSGuard")
    print("🌐 Open your browser to: http://localhost:3002")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=3002)
