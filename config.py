#!/usr/bin/env python3
"""
DriverGuard AI Configuration Management

Centralized configuration management for the DriverGuard AI system.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class Config:
    """Configuration management class."""
    
    def __init__(self, config_path: str = 'configs/driver_guard.yml'):
        """Initialize configuration."""
        self.config_path = config_path
        self.config = self._load_default_config()
        self._load_config_file()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration values."""
        return {
            'detection': {
                'eye_ar_threshold': 0.33,
                'mouth_ar_threshold': 0.7,
                'head_angle_min': 75,
                'head_angle_max': 110,
                'consecutive_frames': 6,
                'detection_mode': '2d_sparse',
                'use_onnx': False,
                'gpu_mode': False
            },
            'web': {
                'host': '0.0.0.0',
                'port': 5000,
                'debug': False,
                'max_alerts': 50,
                'stats_update_interval': 1.0
            },
            'models': {
                'face_detector': {
                    'config_path': 'configs/mb1_120x120.yml',
                    'predictor_path': './shape_predictor_81_face_landmarks (1).dat'
                },
                'yolo': {
                    'enabled': False,
                    'weights_path': 'models/yolov5m.pt',
                    'confidence_threshold': 0.5
                }
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'logs/driver_guard.log'
            }
        }
    
    def _load_config_file(self):
        """Load configuration from file if it exists."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    self._merge_config(file_config)
                logger.info(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config file {self.config_path}: {e}")
        else:
            logger.info(f"Config file {self.config_path} not found, using defaults")
            self._create_default_config_file()
    
    def _merge_config(self, file_config: Dict[str, Any]):
        """Merge file configuration with default configuration."""
        def merge_dicts(default: Dict, override: Dict) -> Dict:
            for key, value in override.items():
                if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                    merge_dicts(default[key], value)
                else:
                    default[key] = value
            return default
        
        self.config = merge_dicts(self.config, file_config)
    
    def _create_default_config_file(self):
        """Create default configuration file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"Default configuration file created at {self.config_path}")
        except Exception as e:
            logger.warning(f"Failed to create config file: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key path."""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """Set configuration value by dot-separated key path."""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save(self):
        """Save current configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def get_detection_config(self) -> Dict[str, Any]:
        """Get detection configuration."""
        return self.get('detection', {})
    
    
    def get_web_config(self) -> Dict[str, Any]:
        """Get web configuration."""
        return self.get('web', {})
    
    def get_models_config(self) -> Dict[str, Any]:
        """Get models configuration."""
        return self.get('models', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get('logging', {})
    
    def setup_logging(self):
        """Setup logging based on configuration."""
        log_config = self.get_logging_config()
        
        # Create logs directory
        log_file = log_config.get('file', 'logs/driver_guard.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logger.info("Logging configured successfully")

# Global configuration instance - removed to avoid conflicts
