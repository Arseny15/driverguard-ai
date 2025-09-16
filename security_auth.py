#!/usr/bin/env python3
"""
DriveGuard AI - Security and Authentication Module

Handles user authentication, session management, and security features.
"""

import hashlib
import secrets
import time
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, session, redirect, url_for
import logging

logger = logging.getLogger(__name__)

class SecurityManager:
    """Manages authentication and security for DriveGuard AI."""
    
    def __init__(self):
        # Default passcode: ARSGuard
        self.default_passcode = "ARSGuard"
        self.session_timeout = 3600  # 1 hour
        self.active_sessions = {}
        self.max_login_attempts = 5
        self.lockout_duration = 300  # 5 minutes
        self.failed_attempts = {}
        
    def hash_password(self, password):
        """Hash password using SHA-256 with salt."""
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}:{password_hash}"
    
    def verify_password(self, password, stored_hash):
        """Verify password against stored hash."""
        try:
            salt, password_hash = stored_hash.split(':')
            return hashlib.sha256((password + salt).encode()).hexdigest() == password_hash
        except:
            return False
    
    def authenticate(self, password):
        """Authenticate user with password."""
        # Check for lockout
        client_ip = request.remote_addr
        if self._is_locked_out(client_ip):
            return False, "Account temporarily locked due to too many failed attempts"
        
        # Verify password
        if password == self.default_passcode:
            # Reset failed attempts on successful login
            if client_ip in self.failed_attempts:
                del self.failed_attempts[client_ip]
            
            # Create session token
            token = self._create_session_token()
            return True, token
        else:
            # Record failed attempt
            self._record_failed_attempt(client_ip)
            return False, "Invalid access code"
    
    def _create_session_token(self):
        """Create a new session token."""
        token = secrets.token_urlsafe(32)
        self.active_sessions[token] = {
            'created_at': time.time(),
            'last_activity': time.time(),
            'ip_address': request.remote_addr
        }
        return token
    
    def _record_failed_attempt(self, client_ip):
        """Record a failed login attempt."""
        if client_ip not in self.failed_attempts:
            self.failed_attempts[client_ip] = {'count': 0, 'last_attempt': 0}
        
        now = time.time()
        attempts = self.failed_attempts[client_ip]
        
        # Reset count if enough time has passed
        if now - attempts['last_attempt'] > self.lockout_duration:
            attempts['count'] = 0
        
        attempts['count'] += 1
        attempts['last_attempt'] = now
        
        logger.warning(f"Failed login attempt from {client_ip} (attempt {attempts['count']})")
    
    def _is_locked_out(self, client_ip):
        """Check if IP is locked out."""
        if client_ip not in self.failed_attempts:
            return False
        
        attempts = self.failed_attempts[client_ip]
        now = time.time()
        
        # Check if still in lockout period
        if now - attempts['last_attempt'] < self.lockout_duration:
            return attempts['count'] >= self.max_login_attempts
        
        return False
    
    def validate_session(self, token):
        """Validate session token."""
        if not token or token not in self.active_sessions:
            return False
        
        session_data = self.active_sessions[token]
        now = time.time()
        
        # Check if session has expired
        if now - session_data['last_activity'] > self.session_timeout:
            del self.active_sessions[token]
            return False
        
        # Update last activity
        session_data['last_activity'] = now
        return True
    
    def logout(self, token):
        """Logout user and invalidate session."""
        if token in self.active_sessions:
            del self.active_sessions[token]
            logger.info(f"User logged out from {request.remote_addr}")
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        now = time.time()
        expired_tokens = []
        
        for token, session_data in self.active_sessions.items():
            if now - session_data['last_activity'] > self.session_timeout:
                expired_tokens.append(token)
        
        for token in expired_tokens:
            del self.active_sessions[token]
        
        if expired_tokens:
            logger.info(f"Cleaned up {len(expired_tokens)} expired sessions")

# Global security manager instance
security_manager = SecurityManager()

def require_auth(f):
    """Decorator to require authentication for routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check for token in Authorization header or session
        token = None
        
        # Try to get token from Authorization header
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        
        # Try to get token from session
        if not token:
            token = session.get('auth_token')
        
        # Try to get token from request args (for API calls)
        if not token:
            token = request.args.get('token')
        
        if not token or not security_manager.validate_session(token):
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Authentication required', 'success': False}), 401
            else:
                return redirect(url_for('login'))
        
        return f(*args, **kwargs)
    return decorated_function

def get_auth_routes(app, security_manager):
    """Add authentication routes to Flask app."""
    from flask import render_template
    
    @app.route('/login')
    def login():
        """Login page."""
        return render_template('login.html')
    
    @app.route('/api/auth', methods=['POST'])
    def authenticate():
        """Authentication API endpoint."""
        try:
            data = request.get_json()
            password = data.get('password', '')
            
            success, result = security_manager.authenticate(password)
            
            if success:
                # Store token in session
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
            logger.error(f"Authentication error: {e}")
            return jsonify({
                'success': False,
                'message': 'Authentication failed'
            }), 500
    
    @app.route('/api/logout', methods=['POST'])
    @require_auth
    def logout():
        """Logout API endpoint."""
        token = session.get('auth_token') or request.headers.get('Authorization', '').replace('Bearer ', '')
        security_manager.logout(token)
        session.pop('auth_token', None)
        return jsonify({'success': True, 'message': 'Logged out successfully'})
    
    @app.route('/api/auth/status')
    def auth_status():
        """Check authentication status."""
        token = session.get('auth_token')
        if token and security_manager.validate_session(token):
            return jsonify({'authenticated': True})
        else:
            return jsonify({'authenticated': False})
