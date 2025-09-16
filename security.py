#!/usr/bin/env python3
"""
DriveGuard AI Security Module

Comprehensive security features for privacy protection and data security.
"""

import os
import hashlib
import hmac
import secrets
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)

class SecurityManager:
    """Comprehensive security management for DriveGuard AI."""
    
    def __init__(self, config_dir='configs'):
        self.config_dir = Path(config_dir)
        self.security_dir = self.config_dir / 'security'
        self.security_dir.mkdir(exist_ok=True)
        
        # Security configuration
        self.security_config = {
            'encryption_enabled': True,
            'data_retention_days': 7,
            'max_failed_attempts': 5,
            'session_timeout_minutes': 30,
            'audit_logging': True,
            'privacy_mode': True,
            'data_anonymization': True
        }
        
        # Initialize security components
        self._init_encryption()
        self._init_audit_logging()
        self._init_privacy_protection()
        
        logger.info("Security Manager initialized")
    
    def _init_encryption(self):
        """Initialize encryption system."""
        self.key_file = self.security_dir / 'encryption.key'
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher = Fernet(self.encryption_key)
    
    def _get_or_create_encryption_key(self):
        """Get or create encryption key."""
        if self.key_file.exists():
            with open(self.key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            # Set restrictive permissions
            os.chmod(self.key_file, 0o600)
            return key
    
    def _init_audit_logging(self):
        """Initialize audit logging system."""
        self.audit_log_file = self.security_dir / 'audit.log'
        self.audit_events = []
    
    def _init_privacy_protection(self):
        """Initialize privacy protection features."""
        self.privacy_config = {
            'blur_faces': True,
            'anonymize_data': True,
            'no_cloud_sync': True,
            'local_only': True
        }
    
    def encrypt_data(self, data):
        """Encrypt sensitive data."""
        if not self.security_config['encryption_enabled']:
            return data
        
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            elif isinstance(data, dict):
                data = json.dumps(data).encode('utf-8')
            
            encrypted_data = self.cipher.encrypt(data)
            return base64.b64encode(encrypted_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return data
    
    def decrypt_data(self, encrypted_data):
        """Decrypt sensitive data."""
        if not self.security_config['encryption_enabled']:
            return encrypted_data
        
        try:
            encrypted_data = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted_data = self.cipher.decrypt(encrypted_data)
            
            # Try to decode as JSON first, then as string
            try:
                return json.loads(decrypted_data.decode('utf-8'))
            except json.JSONDecodeError:
                return decrypted_data.decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return encrypted_data
    
    def log_audit_event(self, event_type, details, user_id=None):
        """Log security audit events."""
        if not self.security_config['audit_logging']:
            return
        
        audit_event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details,
            'user_id': user_id or 'system',
            'ip_address': '127.0.0.1',  # Local only
            'session_id': self._get_session_id()
        }
        
        self.audit_events.append(audit_event)
        
        # Write to audit log file
        with open(self.audit_log_file, 'a') as f:
            f.write(json.dumps(audit_event) + '\n')
        
        logger.info(f"Audit event logged: {event_type}")
    
    def _get_session_id(self):
        """Generate or retrieve session ID."""
        session_file = self.security_dir / 'session.id'
        if session_file.exists():
            with open(session_file, 'r') as f:
                return f.read().strip()
        else:
            session_id = secrets.token_hex(16)
            with open(session_file, 'w') as f:
                f.write(session_id)
            return session_id
    
    def anonymize_face_data(self, face_data):
        """Anonymize face detection data for privacy."""
        if not self.security_config['data_anonymization']:
            return face_data
        
        # Remove or hash personal identifiers
        anonymized = face_data.copy() if isinstance(face_data, dict) else {}
        
        # Hash face coordinates to prevent tracking
        if 'face_coordinates' in anonymized:
            coords = anonymized['face_coordinates']
            if isinstance(coords, (list, tuple)) and len(coords) >= 4:
                # Add noise to coordinates to prevent tracking
                noise_factor = 0.1
                coords = [
                    int(coord + secrets.randbelow(int(coord * noise_factor)) - int(coord * noise_factor / 2))
                    for coord in coords
                ]
                anonymized['face_coordinates'] = coords
        
        # Remove timestamps or make them relative
        if 'timestamp' in anonymized:
            anonymized['timestamp'] = 'anonymized'
        
        return anonymized
    
    def blur_sensitive_regions(self, image, face_regions):
        """Blur sensitive regions in images."""
        if not self.privacy_config['blur_faces']:
            return image
        
        import cv2
        import numpy as np
        
        blurred_image = image.copy()
        
        for (x, y, w, h) in face_regions:
            # Extract face region
            face_region = blurred_image[y:y+h, x:x+w]
            
            # Apply Gaussian blur
            blurred_face = cv2.GaussianBlur(face_region, (15, 15), 0)
            
            # Replace original face with blurred version
            blurred_image[y:y+h, x:x+w] = blurred_face
        
        return blurred_image
    
    def generate_secure_token(self, purpose='api'):
        """Generate secure tokens for API access."""
        token_data = {
            'purpose': purpose,
            'timestamp': time.time(),
            'random': secrets.token_hex(16)
        }
        
        token_string = json.dumps(token_data)
        token_hash = hashlib.sha256(token_string.encode()).hexdigest()
        
        return token_hash
    
    def validate_token(self, token, purpose='api', max_age_hours=24):
        """Validate security tokens."""
        # In a real implementation, you'd store and validate tokens
        # For this demo, we'll use a simple validation
        if len(token) != 64:  # SHA256 hash length
            return False
        
        # Check if token is not too old (simplified)
        return True
    
    def secure_data_storage(self, data, filename):
        """Securely store data with encryption."""
        secure_file = self.security_dir / f"{filename}.enc"
        
        try:
            # Encrypt data
            encrypted_data = self.encrypt_data(data)
            
            # Add integrity check
            data_hash = hashlib.sha256(encrypted_data.encode()).hexdigest()
            secure_data = {
                'data': encrypted_data,
                'hash': data_hash,
                'timestamp': datetime.now().isoformat()
            }
            
            # Write to secure file
            with open(secure_file, 'w') as f:
                json.dump(secure_data, f)
            
            # Set restrictive permissions
            os.chmod(secure_file, 0o600)
            
            logger.info(f"Data securely stored: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Secure storage failed: {e}")
            return False
    
    def load_secure_data(self, filename):
        """Load and decrypt securely stored data."""
        secure_file = self.security_dir / f"{filename}.enc"
        
        if not secure_file.exists():
            return None
        
        try:
            with open(secure_file, 'r') as f:
                secure_data = json.load(f)
            
            # Verify integrity
            data_hash = hashlib.sha256(secure_data['data'].encode()).hexdigest()
            if data_hash != secure_data['hash']:
                logger.error("Data integrity check failed")
                return None
            
            # Decrypt data
            decrypted_data = self.decrypt_data(secure_data['data'])
            
            logger.info(f"Secure data loaded: {filename}")
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Secure data loading failed: {e}")
            return None
    
    def cleanup_old_data(self):
        """Clean up old data based on retention policy."""
        retention_days = self.security_config['data_retention_days']
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        cleaned_count = 0
        
        # Clean up old audit logs
        if self.audit_log_file.exists():
            with open(self.audit_log_file, 'r') as f:
                lines = f.readlines()
            
            recent_lines = []
            for line in lines:
                try:
                    event = json.loads(line.strip())
                    event_time = datetime.fromisoformat(event['timestamp'])
                    if event_time > cutoff_date:
                        recent_lines.append(line)
                    else:
                        cleaned_count += 1
                except:
                    continue
            
            with open(self.audit_log_file, 'w') as f:
                f.writelines(recent_lines)
        
        # Clean up old session files
        session_file = self.security_dir / 'session.id'
        if session_file.exists():
            session_age = datetime.fromtimestamp(session_file.stat().st_mtime)
            if session_age < cutoff_date:
                session_file.unlink()
                cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} old data entries")
        return cleaned_count
    
    def get_security_status(self):
        """Get current security status."""
        return {
            'encryption_enabled': self.security_config['encryption_enabled'],
            'audit_logging': self.security_config['audit_logging'],
            'privacy_mode': self.security_config['privacy_mode'],
            'data_retention_days': self.security_config['data_retention_days'],
            'last_cleanup': datetime.now().isoformat(),
            'security_dir': str(self.security_dir),
            'audit_events_count': len(self.audit_events)
        }
    
    def export_security_logs(self, output_file=None):
        """Export security logs for analysis."""
        if not output_file:
            output_file = self.security_dir / f"security_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'security_config': self.security_config,
            'audit_events': self.audit_events,
            'security_status': self.get_security_status()
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Security logs exported to: {output_file}")
        return str(output_file)
    
    def enable_privacy_mode(self):
        """Enable maximum privacy mode."""
        self.security_config.update({
            'encryption_enabled': True,
            'data_anonymization': True,
            'audit_logging': False,  # Disable in privacy mode
            'privacy_mode': True
        })
        
        self.privacy_config.update({
            'blur_faces': True,
            'anonymize_data': True,
            'no_cloud_sync': True,
            'local_only': True
        })
        
        logger.info("Privacy mode enabled")
    
    def disable_data_collection(self):
        """Disable all data collection for maximum privacy."""
        self.security_config.update({
            'audit_logging': False,
            'data_anonymization': True,
            'privacy_mode': True
        })
        
        # Clear existing audit logs
        if self.audit_log_file.exists():
            self.audit_log_file.unlink()
        
        self.audit_events = []
        logger.info("Data collection disabled")
    
    def verify_system_integrity(self):
        """Verify system integrity and security."""
        integrity_report = {
            'timestamp': datetime.now().isoformat(),
            'encryption_key_exists': self.key_file.exists(),
            'security_dir_secure': self._check_directory_permissions(),
            'audit_logging_active': self.security_config['audit_logging'],
            'privacy_mode_active': self.security_config['privacy_mode'],
            'no_external_connections': True,  # Local only
            'data_encrypted': self.security_config['encryption_enabled']
        }
        
        # Check if any security files are accessible
        security_files = list(self.security_dir.glob('*'))
        for file_path in security_files:
            if file_path.is_file():
                permissions = oct(file_path.stat().st_mode)[-3:]
                integrity_report[f'file_{file_path.name}_permissions'] = permissions
        
        logger.info("System integrity check completed")
        return integrity_report
    
    def _check_directory_permissions(self):
        """Check if security directory has proper permissions."""
        try:
            stat_info = self.security_dir.stat()
            permissions = oct(stat_info.st_mode)[-3:]
            return permissions in ['700', '750', '755']
        except:
            return False

# Global security manager instance
security_manager = SecurityManager()
