#!/usr/bin/env python3
"""
DriveGuard AI - Command Line Demo

A simple command-line version for testing without browser.
"""

import cv2
import time
import numpy as np

def main():
    print("🚗 DriveGuard AI - Command Line Demo")
    print("📹 Starting webcam...")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Cannot open webcam")
        return
    
    print("✅ Webcam opened successfully")
    print("👁️  Press 'q' to quit, 's' to start/stop detection")
    
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    detection_active = False
    frame_count = 0
    face_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if detection_active:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                face_count += 1
            
            # Add status information
            cv2.putText(frame, f'Detection: ACTIVE', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'Frames: {frame_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'Faces: {face_count}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f'Detection: INACTIVE', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f'Frames: {frame_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f'Faces: {face_count}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show frame
        cv2.imshow('DriveGuard AI - CLI Demo', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            detection_active = not detection_active
            status = "ACTIVE" if detection_active else "INACTIVE"
            print(f"🔍 Detection {status}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 DriveGuard AI demo ended")

if __name__ == '__main__':
    main()
