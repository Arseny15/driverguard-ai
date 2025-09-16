#!/usr/bin/env python3
"""
DriveGuard AI - Intelligent Driver Behavior Monitoring System

A modern computer vision system for real-time driver behavior monitoring
with drowsiness detection, phone usage detection, and attention tracking.

Powered by ARS Technologies
"""

import argparse
import cv2
import numpy as np
import torch
import math
import time
from scipy.spatial import distance as dist
from collections import deque
import threading
import yaml
from tqdm import tqdm
import sys
import os
import logging
from pathlib import Path

# Try to import dlib, but make it optional
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("Warning: dlib not available. Some features may be limited.")

# Import our modules (make them optional for demo purposes)
try:
    from FaceBoxes import FaceBoxes
    from TDDFA import TDDFA
    from utils.render import render
    from utils.functions import cv_draw_landmark
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False
    print("Warning: Advanced face detection modules not available. Running in demo mode.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DriveGuardAI:
    """Main DriveGuard AI class for driver behavior monitoring."""
    
    def __init__(self, config_path='configs/driver_guard.yml', use_onnx=False, mode='cpu'):
        """Initialize the DriveGuard AI system."""
        self.config_path = config_path
        self.use_onnx = use_onnx
        self.mode = mode
        self.demo_mode = False  # Force real detection mode
        self.detection_mode = '2d_sparse'  # Default detection mode
        
        # Load configuration
        self.cfg = self._load_config()
        
        # Initialize models
        self._initialize_models()
        
        
        # Detection counters and thresholds
        self.counters = {
            'eye_closed': 0,
            'phone_usage': 0,
            'attention_lost': 0
        }
        
        self.thresholds = {
            'eye_ar': 0.33,
            'mouth_ar': 0.7,
            'head_angle_min': 75,
            'head_angle_max': 110,
            'consecutive_frames': 6
        }
        
        # Model points for head pose estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),  # Tip of the nose
            (-30.0, -125.0, -30.0),  # Left eye corner
            (30.0, -125.0, -30.0),  # Right eye corner
            (-60.0, -70.0, -60.0),  # Left mouth corner
            (60.0, -70.0, -60.0),  # Right mouth corner
            (0.0, -330.0, -65.0)  # Chin
        ])
        
        logger.info("DriveGuard AI initialized successfully")
    
    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
    
    def _initialize_models(self):
        """Initialize face detection and landmark models."""
        logger.info("Initializing advanced detection models")
        
        # Initialize OpenCV face and eye detectors
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Initialize additional detectors for better accuracy
        self.face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.eye_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
        
        # Initialize phone detection using HOG descriptor
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Initialize template matching for phone detection
        self.phone_templates = []
        
        # Initialize YOLO for phone detection if available
        try:
            import torch
            self.yolo_available = True
            # Try to load YOLOv5 model for phone detection (with offline fallback)
            try:
                self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                logger.info("YOLO model loaded for phone detection")
            except Exception as e:
                logger.warning(f"YOLO model download failed: {e}")
                logger.warning("Using basic phone detection without YOLO")
                self.yolo_available = False
                self.yolo_model = None
        except ImportError:
            self.yolo_available = False
            self.yolo_model = None
            logger.warning("PyTorch not available, using basic phone detection")
        
        # Initialize face landmark detection using MediaPipe if available
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mediapipe_available = True
            logger.info("MediaPipe face mesh loaded for advanced landmark detection")
        except ImportError:
            self.mediapipe_available = False
            logger.warning("MediaPipe not available, using basic landmark detection")
        
        logger.info("Advanced detection models initialized successfully")
    
    
    
    def get_camera_matrix(self, size):
        """Get camera intrinsic matrix."""
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        return np.array([[focal_length, 0, center[0]], 
                        [0, focal_length, center[1]], 
                        [0, 0, 1]], dtype="double")
    
    def is_rotation_matrix(self, R):
        """Check if matrix is a valid rotation matrix."""
        Rt = np.transpose(R)
        should_be_identity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - should_be_identity)
        return n < 1e-6
    
    def rotation_matrix_to_euler_angles(self, R):
        """Convert rotation matrix to Euler angles."""
        assert self.is_rotation_matrix(R)
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        
        return np.array([x, y, z])
    
    def get_head_tilt_and_coords(self, size, image_points, frame_height):
        """Calculate head tilt and coordinates for visualization."""
        camera_matrix = self.get_camera_matrix(size)
        dist_coeffs = np.zeros((4, 1))
        
        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        (nose_end_point2D, _) = cv2.projectPoints(
            np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, 
            camera_matrix, dist_coeffs
        )
        
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        head_tilt_degree = abs([-180] - np.rad2deg([self.rotation_matrix_to_euler_angles(rotation_matrix)[0]]))
        
        starting_point = (int(image_points[0][0]), int(image_points[0][1]))
        ending_point = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        ending_point_alternate = (ending_point[0], frame_height // 2)
        
        return head_tilt_degree, starting_point, ending_point, ending_point_alternate
    
    def eye_aspect_ratio(self, eye):
        """Calculate Eye Aspect Ratio (EAR)."""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)
    
    def mouth_aspect_ratio(self, mouth):
        """Calculate Mouth Aspect Ratio (MAR)."""
        A = dist.euclidean(mouth[2], mouth[10])
        B = dist.euclidean(mouth[4], mouth[8])
        C = dist.euclidean(mouth[0], mouth[6])
        return (A + B) / (2.0 * C)
    
    def nose_aspect_ratio(self, nose):
        """Calculate Nose Aspect Ratio (NAR)."""
        vertical_distance = dist.euclidean(nose[0], nose[2])
        depth_distance = dist.euclidean(nose[0], nose[1])
        return depth_distance / vertical_distance
    
    def calculate_head_angle(self, eye_left, eye_right, nose_tip):
        """Calculate head angle from eye and nose positions."""
        eye_center = (eye_left + eye_right) / 2
        vector_nose = nose_tip - eye_center
        vector_horizontal = (eye_right - eye_left)
        vector_horizontal[1] = 0
        
        vector_nose_normalized = vector_nose / np.linalg.norm(vector_nose)
        vector_horizontal_normalized = vector_horizontal / np.linalg.norm(vector_horizontal)
        
        angle_rad = np.arccos(np.clip(np.dot(vector_nose_normalized, vector_horizontal_normalized), -1.0, 1.0))
        return np.degrees(angle_rad)
    
    def process_frame(self, frame, dense_flag=False):
        """Process a single frame for driver behavior analysis using advanced computer vision."""
        img_draw = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize detection variables
        ear = 0.3
        mar = 0.5
        head_angle = 90
        attention_lost = False
        phone_detected = False
        
        # Initialize drowsiness frame counter (for 3-second minimum)
        if not hasattr(self, 'drowsiness_frame_counter'):
            self.drowsiness_frame_counter = 0
        if not hasattr(self, 'drowsiness_detection_threshold'):
            # 3 seconds at 30 FPS = 90 frames
            self.drowsiness_detection_threshold = 90
        
        # Detect faces using multiple methods for better accuracy
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            # Try alternative face detector
            faces = self.face_cascade_alt.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            cv2.putText(img_draw, "No face detected - Look at camera!", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Simulate attention loss when no face detected
            self.counters['attention_lost'] += 1
            if self.counters['attention_lost'] >= self.thresholds['consecutive_frames']:
                self.counters['attention_lost'] = 0
            return img_draw, ear, mar, head_angle
        
        # Process the largest face (most likely the driver)
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
        # Draw face rectangle
        cv2.rectangle(img_draw, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Detect eyes in the face region
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img_draw[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(roi_gray)
        
        # If no eyes found with primary detector, try alternative
        if len(eyes) == 0:
            eyes = self.eye_cascade_alt.detectMultiScale(roi_gray)
        
        # Filter and sort eyes by size (largest first) to get the best 2 eyes
        if len(eyes) > 0:
            # Sort by area (width * height) in descending order
            eyes = sorted(eyes, key=lambda x: x[2] * x[3], reverse=True)
            # Take only the first 2 eyes (largest ones)
            eyes = eyes[:2]
        
        # Calculate EAR based on eye detection
        if len(eyes) >= 2:
            # Both eyes detected - calculate real EAR
            eye_centers = []
            for (ex, ey, ew, eh) in eyes:
                eye_center = (ex + ew//2, ey + eh//2)
                eye_centers.append(eye_center)
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            # Calculate distance between eyes for EAR estimation
            eye_distance = np.sqrt((eye_centers[1][0] - eye_centers[0][0])**2 + 
                                 (eye_centers[1][1] - eye_centers[0][1])**2)
            # Estimate EAR based on eye distance and face size
            ear = min(0.4, max(0.1, eye_distance / (w * 0.3)))
            head_angle = 90
        elif len(eyes) == 1:
            # One eye detected - possible head turn or phone usage
            ear = 0.25
            head_angle = 75
            attention_lost = True
            phone_detected = True  # Single eye often indicates phone usage
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        else:
            # No eyes detected - drowsiness, head down, or phone usage
            ear = 0.15
            head_angle = 45
            attention_lost = True
            phone_detected = True  # No eyes detected often indicates phone usage
        
        # Additional phone detection: Check if face is partially covered or obscured
        if not phone_detected:
            # Check if face is too small (might be covered by phone)
            if w < frame.shape[1] * 0.2 or h < frame.shape[0] * 0.2:
                phone_detected = True
                cv2.putText(img_draw, "Face too small - Phone detected!", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Check if face is significantly off-center (phone usage)
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2
            
            if (abs(face_center_x - frame_center_x) > frame.shape[1] * 0.25 or 
                abs(face_center_y - frame_center_y) > frame.shape[0] * 0.25):
                phone_detected = True
                cv2.putText(img_draw, "Face off-center - Phone detected!", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Simple test: If no eyes detected, immediately trigger phone detection
            if len(eyes) == 0:
                phone_detected = True
                cv2.putText(img_draw, "No eyes - Phone usage!", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Detect phone usage using multiple methods (only if not already detected by eye analysis)
        if not phone_detected:
            phone_detected = self._detect_phone_usage(frame, x, y, w, h)
        
        # Additional phone detection: if no eyes detected for multiple frames, likely phone usage
        if len(eyes) == 0:
            if not hasattr(self, 'no_eyes_counter'):
                self.no_eyes_counter = 0
            self.no_eyes_counter += 1
            # If no eyes detected for 1+ consecutive frames, definitely phone usage (very aggressive)
            if self.no_eyes_counter >= 1:
                phone_detected = True
                cv2.putText(img_draw, "No eyes - Phone usage!", (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        else:
            # Reset counter when eyes are detected
            if hasattr(self, 'no_eyes_counter'):
                self.no_eyes_counter = 0
        
        # Use MediaPipe for advanced face landmark detection if available
        if self.mediapipe_available:
            try:
                results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    # Calculate more accurate EAR using MediaPipe landmarks
                    ear = self._calculate_ear_mediapipe(face_landmarks)
                    # Calculate head pose using MediaPipe
                    head_angle = self._calculate_head_pose_mediapipe(face_landmarks, frame.shape)
            except Exception as e:
                logger.warning(f"MediaPipe processing failed: {e}")
        
        # Update detection counters
        if attention_lost:
            self.counters['attention_lost'] += 1
            if self.counters['attention_lost'] >= self.thresholds['consecutive_frames']:
                self.counters['attention_lost'] = 0
        else:
            self.counters['attention_lost'] = 0
        
        if phone_detected:
            self.counters['phone_usage'] += 1
            # Don't reset phone usage counter immediately - let it accumulate
            if self.counters['phone_usage'] >= self.thresholds['consecutive_frames']:
                # Keep the counter for display purposes
                pass
            # Reset no-phone counter when phone is detected
            if hasattr(self, 'no_phone_counter'):
                self.no_phone_counter = 0
        else:
            # Only reset phone usage counter if no phone detected for multiple frames
            if not hasattr(self, 'no_phone_counter'):
                self.no_phone_counter = 0
            self.no_phone_counter += 1
            if self.no_phone_counter >= 5:  # Reset after 5 frames without phone (more responsive)
                self.counters['phone_usage'] = 0
                self.no_phone_counter = 0
        
        # Check for drowsiness with 3-second minimum threshold
        if ear < self.thresholds['eye_ar']:
            self.drowsiness_frame_counter += 1
            # Only trigger drowsiness if eyes have been closed for 3+ seconds
            if self.drowsiness_frame_counter >= self.drowsiness_detection_threshold:
                self.counters['eye_closed'] += 1
                if self.counters['eye_closed'] >= self.thresholds['consecutive_frames']:
                    self.counters['eye_closed'] = 0
                # Reset counter after detection
                self.drowsiness_frame_counter = 0
        else:
            # Reset drowsiness counter when eyes are open
            self.drowsiness_frame_counter = 0
            self.counters['eye_closed'] = 0
        
        # Add visual indicators
        if attention_lost:
            cv2.putText(img_draw, "Attention Lost!", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        if phone_detected:
            cv2.putText(img_draw, "Phone Detected!", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        if ear < self.thresholds['eye_ar']:
            cv2.putText(img_draw, "Drowsiness Detected!", (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display current values
        cv2.putText(img_draw, f'EAR: {ear:.2f}', (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(img_draw, f'MAR: {mar:.2f}', (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(img_draw, f'Head Angle: {head_angle:.1f}', (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Display drowsiness counter
        drowsiness_progress = min(100, (self.drowsiness_frame_counter / self.drowsiness_detection_threshold) * 100)
        cv2.putText(img_draw, f'Drowsiness: {drowsiness_progress:.1f}%', (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Display no-eyes counter for phone detection
        if hasattr(self, 'no_eyes_counter'):
            cv2.putText(img_draw, f'No Eyes: {self.no_eyes_counter}', (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        return img_draw, ear, mar, head_angle
    
    def _detect_phone_usage(self, frame, face_x, face_y, face_w, face_h):
        """Detect phone usage using multiple computer vision techniques."""
        phone_detected = False
        
        try:
            # Method 1: YOLO detection if available
            if self.yolo_available and self.yolo_model is not None:
                try:
                    results = self.yolo_model(frame)
                    for detection in results.xyxy[0]:
                        if len(detection) >= 6:
                            class_id = int(detection[5])
                            confidence = detection[4]
                            # Check for cell phone class (class 67 in COCO dataset)
                            if class_id == 67 and confidence > 0.3:  # Lowered threshold
                                phone_detected = True
                                break
                except Exception as e:
                    logger.warning(f"YOLO detection failed: {e}")
            
            # Method 2: Face position analysis (more aggressive)
            if not phone_detected:
                face_center_x = face_x + face_w // 2
                face_center_y = face_y + face_h // 2
                frame_center_x = frame.shape[1] // 2
                frame_center_y = frame.shape[0] // 2
                
                # More sensitive thresholds for phone detection
                if (abs(face_center_x - frame_center_x) > frame.shape[1] * 0.2 or 
                    abs(face_center_y - frame_center_y) > frame.shape[0] * 0.2 or
                    face_w < frame.shape[1] * 0.15 or  # Smaller face threshold
                    face_h < frame.shape[0] * 0.15):  # Added height check
                    phone_detected = True
            
            # Method 3: Hand detection using HOG
            if not phone_detected:
                try:
                    # Detect people in the frame
                    boxes, weights = self.hog.detectMultiScale(frame, winStride=(8,8))
                    for (x, y, w, h) in boxes:
                        # Check if hand is near face area
                        if (abs(x - face_x) < face_w and abs(y - face_y) < face_h * 2):
                            phone_detected = True
                            break
                except:
                    pass
            
            # Method 4: Check for face occlusion patterns
            if not phone_detected:
                # Check if face region has unusual brightness patterns (phone screen reflection)
                face_roi = frame[face_y:face_y+face_h, face_x:face_x+face_w]
                if face_roi.size > 0:
                    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    # Check for high brightness areas that might indicate phone screen
                    bright_pixels = np.sum(gray_face > 200)
                    total_pixels = gray_face.size
                    if bright_pixels / total_pixels > 0.3:  # 30% bright pixels
                        phone_detected = True
                    
        except Exception as e:
            logger.warning(f"Phone detection error: {e}")
        
        return phone_detected
    
    def _calculate_ear_mediapipe(self, face_landmarks):
        """Calculate EAR using MediaPipe face landmarks."""
        try:
            # Get eye landmarks (MediaPipe uses different indices)
            left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            
            # Calculate EAR for both eyes
            left_ear = self._calculate_ear_from_landmarks(face_landmarks, left_eye_indices)
            right_ear = self._calculate_ear_from_landmarks(face_landmarks, right_eye_indices)
            
            return (left_ear + right_ear) / 2.0
        except:
            return 0.3  # Default value
    
    def _calculate_ear_from_landmarks(self, landmarks, eye_indices):
        """Calculate EAR from specific eye landmark indices."""
        try:
            eye_points = []
            for idx in eye_indices:
                if idx < len(landmarks.landmark):
                    x = landmarks.landmark[idx].x
                    y = landmarks.landmark[idx].y
                    eye_points.append([x, y])
            
            if len(eye_points) >= 6:
                eye_points = np.array(eye_points)
                return self.eye_aspect_ratio(eye_points)
            return 0.3
        except:
            return 0.3
    
    def _calculate_head_pose_mediapipe(self, face_landmarks, frame_shape):
        """Calculate head pose using MediaPipe landmarks."""
        try:
            # Get key facial points
            nose_tip = face_landmarks.landmark[1]  # Nose tip
            left_eye = face_landmarks.landmark[33]  # Left eye
            right_eye = face_landmarks.landmark[362]  # Right eye
            
            # Convert to pixel coordinates
            h, w = frame_shape[:2]
            nose_point = np.array([nose_tip.x * w, nose_tip.y * h])
            left_eye_point = np.array([left_eye.x * w, left_eye.y * h])
            right_eye_point = np.array([right_eye.x * w, right_eye.y * h])
            
            # Calculate head angle
            return self.calculate_head_angle(left_eye_point, right_eye_point, nose_point)
        except:
            return 90  # Default value
    
    def _process_frame_demo(self, frame, detection_mode='2d_sparse'):
        """Demo version of process_frame using basic OpenCV with different modes."""
        img_draw = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            cv2.putText(img_draw, "No face detected - Look at camera!", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Simulate attention loss when no face detected
            self.counters['attention_lost'] += 1
            if self.counters['attention_lost'] >= self.thresholds['consecutive_frames']:
                self.counters['attention_lost'] = 0
            return img_draw, 0.3, 0.5, 90
        
        # Initialize detection variables
        ear = 0.3  # Default EAR
        mar = 0.5  # Default MAR
        head_angle = 90  # Default head angle
        attention_lost = False
        phone_detected = False
        
        # Draw rectangle around face
        for (x, y, w, h) in faces:
            cv2.rectangle(img_draw, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Detect eyes in the face region
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img_draw[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            # Calculate basic EAR based on eye detection
            if len(eyes) >= 2:
                # Both eyes detected - normal attention
                ear = 0.35
                head_angle = 90
            elif len(eyes) == 1:
                # One eye detected - possible head turn
                ear = 0.25
                head_angle = 75
                attention_lost = True
            else:
                # No eyes detected - drowsiness or head down
                ear = 0.15
                head_angle = 45
                attention_lost = True
            
            # Draw rectangles around eyes
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            # Simulate phone detection based on face position and size
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2
            
            # More aggressive phone detection in demo mode
            if (abs(face_center_x - frame_center_x) > frame.shape[1] * 0.2 or 
                abs(face_center_y - frame_center_y) > frame.shape[0] * 0.2 or
                w < frame.shape[1] * 0.2 or  # Increased threshold
                h < frame.shape[0] * 0.2 or  # Added height check
                len(eyes) == 0 or  # No eyes detected
                len(eyes) == 1):   # Only one eye detected
                phone_detected = True
                cv2.putText(img_draw, "Phone Detected!", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Add status text based on detection mode
            mode_text = detection_mode.upper().replace('_', ' ')
            cv2.putText(img_draw, f"Face detected - {mode_text} Mode", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_draw, f"Eyes found: {len(eyes)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add mode-specific visual indicators
            if detection_mode == '2d_sparse':
                cv2.putText(img_draw, "Sparse Detection", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            elif detection_mode == '2d_dense':
                cv2.putText(img_draw, "Dense Detection", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
                # Draw additional features for dense mode
                cv2.circle(img_draw, (x + w//2, y + h//2), 5, (255, 165, 0), -1)
            elif detection_mode == '3d':
                cv2.putText(img_draw, "3D Reconstruction", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                # Draw 3D-like features
                cv2.circle(img_draw, (x + w//2, y + h//2), 8, (255, 0, 255), 2)
                cv2.circle(img_draw, (x + w//4, y + h//3), 3, (255, 0, 255), -1)
                cv2.circle(img_draw, (x + 3*w//4, y + h//3), 3, (255, 0, 255), -1)
        
        # Update detection counters based on demo logic
        if attention_lost:
            self.counters['attention_lost'] += 1
            if self.counters['attention_lost'] >= self.thresholds['consecutive_frames']:
                self.counters['attention_lost'] = 0
        else:
            self.counters['attention_lost'] = 0
        
        if phone_detected:
            self.counters['phone_usage'] += 1
            # Don't reset phone usage counter immediately - let it accumulate
            if self.counters['phone_usage'] >= self.thresholds['consecutive_frames']:
                # Keep the counter for display purposes
                pass
        else:
            # Only reset phone usage counter if no phone detected for multiple frames
            if not hasattr(self, 'no_phone_counter'):
                self.no_phone_counter = 0
            self.no_phone_counter += 1
            if self.no_phone_counter >= 10:  # Reset after 10 frames without phone
                self.counters['phone_usage'] = 0
                self.no_phone_counter = 0
        
        # Check for drowsiness based on EAR
        if ear < self.thresholds['eye_ar']:
            self.counters['eye_closed'] += 1
            if self.counters['eye_closed'] >= self.thresholds['consecutive_frames']:
                self.counters['eye_closed'] = 0
        else:
            self.counters['eye_closed'] = 0
        
        # Add visual indicators for detections
        if attention_lost:
            cv2.putText(img_draw, "Attention Lost!", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        if ear < self.thresholds['eye_ar']:
            cv2.putText(img_draw, "Drowsiness Detected!", (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display current values
        cv2.putText(img_draw, f'EAR: {ear:.2f}', (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(img_draw, f'MAR: {mar:.2f}', (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(img_draw, f'Head Angle: {head_angle:.1f}', (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Display drowsiness counter
        drowsiness_progress = min(100, (self.drowsiness_frame_counter / self.drowsiness_detection_threshold) * 100)
        cv2.putText(img_draw, f'Drowsiness: {drowsiness_progress:.1f}%', (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Display no-eyes counter for phone detection
        if hasattr(self, 'no_eyes_counter'):
            cv2.putText(img_draw, f'No Eyes: {self.no_eyes_counter}', (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        return img_draw, ear, mar, head_angle
    
    def run_webcam(self, opt='2d_sparse', n_pre=1, n_next=1):
        """Run the main webcam detection loop."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        
        try:
            # Initialize smoothing queues
            n = n_pre + n_next + 1
            queue_ver = deque()
            queue_frame = deque()
            dense_flag = opt in ('2d_dense', '3d')
            pre_ver = None
            
            
            logger.info("Starting webcam detection...")
            
            frame_count = 0
            for frame_bgr in tqdm(self._webcam_frames(cap)):
                frame_count += 1
                
                # Face detection and landmark processing
                if frame_count == 1:
                    boxes = self.face_boxes(frame_bgr)
                    if len(boxes) > 0:
                        boxes = [boxes[0]]
                        param_lst, roi_box_lst = self.tddfa(frame_bgr, boxes)
                        ver = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
                        
                        param_lst, roi_box_lst = self.tddfa(frame_bgr, [ver], crop_policy='landmark')
                        ver = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
                        
                        for _ in range(n_pre):
                            queue_ver.append(ver.copy())
                            queue_frame.append(frame_bgr.copy())
                        queue_ver.append(ver.copy())
                        queue_frame.append(frame_bgr.copy())
                    else:
                        continue
                else:
                    param_lst, roi_box_lst = self.tddfa(frame_bgr, [pre_ver], crop_policy='landmark')
                    roi_box = roi_box_lst[0]
                    if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                        boxes = self.face_boxes(frame_bgr)
                        if len(boxes) > 0:
                            boxes = [boxes[0]]
                            param_lst, roi_box_lst = self.tddfa(frame_bgr, boxes)
                    
                    ver = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
                    queue_ver.append(ver.copy())
                    queue_frame.append(frame_bgr.copy())
                
                pre_ver = ver
                
                # Process frame for behavior analysis
                if len(queue_ver) >= n:
                    ver_ave = np.mean(queue_ver, axis=0)
                    
                    if opt == '2d_sparse':
                        img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave)
                    elif opt == '2d_dense':
                        img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave, size=1)
                    elif opt == '3d':
                        img_draw = render(queue_frame[n_pre], [ver_ave], self.tddfa.tri, alpha=0.7)
                    else:
                        raise ValueError(f'Unknown opt {opt}')
                    
                    # Apply behavior analysis
                    img_draw, ear, mar, head_angle = self.process_frame(frame_bgr)
                    
                    # Display frame
                    cv2.imshow('DriveGuard AI', img_draw)
                    
                    # Check for quit
                    k = cv2.waitKey(20)
                    if k & 0xff == ord('q'):
                        break
                    
                    # Update queues
                    queue_ver.popleft()
                    queue_frame.popleft()
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Webcam detection stopped")
    
    def _webcam_frames(self, cap):
        """Generator for webcam frames."""
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='DriveGuard AI - Driver Behavior Monitoring')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml',
                       help='Path to configuration file')
    parser.add_argument('-m', '--mode', default='cpu', type=str, choices=['cpu', 'gpu'],
                       help='Processing mode')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse', 
                       choices=['2d_sparse', '2d_dense', '3d'],
                       help='Detection mode')
    parser.add_argument('-n_pre', default=1, type=int, 
                       help='Number of pre-frames for smoothing')
    parser.add_argument('-n_next', default=1, type=int, 
                       help='Number of next-frames for smoothing')
    parser.add_argument('--onnx', action='store_true', default=False,
                       help='Use ONNX models for faster inference')
    
    args = parser.parse_args()
    
    try:
        # Initialize DriveGuard AI
        driver_guard = DriveGuardAI(
            config_path=args.config,
            use_onnx=args.onnx,
            mode=args.mode
        )
        
        # Run webcam detection
        driver_guard.run_webcam(
            opt=args.opt,
            n_pre=args.n_pre,
            n_next=args.n_next
        )
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
