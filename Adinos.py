"""
Advanced Face Detection Auto-Lock System
Uses MediaPipe for superior accuracy and speed with anti-spoofing features
"""

import cv2
import time
import platform
import subprocess
import logging
from datetime import datetime
from collections import deque
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("WARNING: MediaPipe not installed. Install with: pip install mediapipe")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_lock.log'),
        logging.StreamHandler()
    ]
)

class AdvancedFaceDetectionLock:
    def __init__(self, 
                 check_interval=2,
                 no_face_threshold=10,
                 min_detection_confidence=0.7,
                 enable_liveness=True,
                 enable_logging=True):
        """
        Advanced face detection with MediaPipe and anti-spoofing.
        
        Args:
            check_interval: Seconds between face detection checks
            no_face_threshold: Seconds without face before locking
            min_detection_confidence: Detection confidence threshold (0.5-1.0)
            enable_liveness: Enable basic liveness detection
            enable_logging: Enable detailed logging
        """
        self.check_interval = check_interval
        self.no_face_threshold = no_face_threshold
        self.min_detection_confidence = min_detection_confidence
        self.enable_liveness = enable_liveness
        self.enable_logging = enable_logging
        
        self.no_face_time = 0
        self.last_check_time = time.time()
        self.lock_count = 0
        self.detection_history = deque(maxlen=5)  # Track last 5 detections
        
        # Initialize face detection
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,  # 0 for short-range (< 2m), 1 for full-range
                min_detection_confidence=min_detection_confidence
            )
            self.detector_type = "MediaPipe"
            logging.info("Initialized MediaPipe face detector")
        else:
            # Fallback to Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            self.detector_type = "Haar Cascade (Fallback)"
            logging.warning("MediaPipe unavailable, using Haar Cascade fallback")
        
        # Initialize camera
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise RuntimeError("Could not open camera")
        
        # Set camera properties for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Detect operating system
        self.os_type = platform.system()
        logging.info(f"Running on: {self.os_type}")
        logging.info(f"Detector: {self.detector_type}")
    
    def detect_face_mediapipe(self, frame):
        """Detect faces using MediaPipe (faster and more accurate)."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            face_count = len(results.detections)
            confidences = [detection.score[0] for detection in results.detections]
            avg_confidence = sum(confidences) / len(confidences)
            
            return True, face_count, avg_confidence
        return False, 0, 0.0
    
    def detect_face_haar(self, frame):
        """Fallback: Detect faces using Haar Cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        
        if len(faces) > 0:
            return True, len(faces), 0.8  # Simulated confidence
        return False, 0, 0.0
    
    def check_liveness(self, frame):
        """
        Basic liveness check using motion detection.
        More advanced methods would use blink detection, 3D depth, etc.
        """
        if not hasattr(self, 'prev_frame'):
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return True
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        motion_score = np.mean(frame_diff)
        
        self.prev_frame = gray
        
        # If there's some motion (not a static photo), consider it live
        # Threshold: 5-50 suggests subtle natural movement
        is_live = 5 < motion_score < 50
        
        return is_live
    
    def detect_face(self):
        """Check if a face is present with enhanced detection."""
        ret, frame = self.camera.read()
        if not ret:
            logging.error("Failed to grab frame")
            return False, {}
        
        # Detect face
        if MEDIAPIPE_AVAILABLE:
            face_detected, face_count, confidence = self.detect_face_mediapipe(frame)
        else:
            face_detected, face_count, confidence = self.detect_face_haar(frame)
        
        # Liveness check
        is_live = True
        if self.enable_liveness and face_detected:
            is_live = self.check_liveness(frame)
        
        # Store detection in history
        self.detection_history.append(face_detected and is_live)
        
        # Require majority of recent detections to be positive (reduces false negatives)
        stable_detection = sum(self.detection_history) >= len(self.detection_history) // 2
        
        info = {
            'face_count': face_count,
            'confidence': confidence,
            'is_live': is_live,
            'stable': stable_detection
        }
        
        return stable_detection, info
    
    def lock_computer(self):
        """Lock the computer based on the operating system."""
        self.lock_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.warning(f"LOCKING COMPUTER #{self.lock_count} at {timestamp}")
        
        try:
            if self.os_type == "Windows":
                subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"])
            elif self.os_type == "Darwin":  # macOS
                subprocess.run(["/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession", "-suspend"])
            elif self.os_type == "Linux":
                # Try multiple Linux lock commands
                commands = [
                    ["gnome-screensaver-command", "--lock"],
                    ["xdg-screensaver", "lock"],
                    ["loginctl", "lock-session"],
                    ["dm-tool", "lock"]
                ]
                locked = False
                for cmd in commands:
                    try:
                        subprocess.run(cmd, check=True, timeout=2)
                        locked = True
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                        continue
                
                if not locked:
                    logging.error("Failed to lock computer - no working lock command found")
                    return
            
            logging.info("Computer locked successfully")
        except Exception as e:
            logging.error(f"Error locking computer: {e}")
    
    def run(self):
        """Main loop for face detection and auto-lock."""
        print("\n" + "="*60)
        print("🔒 Advanced Face Detection Auto-Lock System")
        print("="*60)
        print(f"Detector: {self.detector_type}")
        print(f"Check interval: {self.check_interval}s")
        print(f"Lock threshold: {self.no_face_threshold}s")
        print(f"Liveness detection: {'Enabled' if self.enable_liveness else 'Disabled'}")
        print(f"Min confidence: {self.min_detection_confidence}")
        print(f"Logging: {'face_lock.log' if self.enable_logging else 'Disabled'}")
        print("\n💡 Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        consecutive_failures = 0
        
        try:
            while True:
                current_time = time.time()
                
                # Check if it's time for another detection
                if current_time - self.last_check_time >= self.check_interval:
                    try:
                        face_detected, info = self.detect_face()
                        
                        if face_detected:
                            if self.no_face_time > 0:
                                msg = f"✅ Face detected (conf: {info['confidence']:.2f}, "
                                msg += f"live: {info['is_live']}) - Reset timer"
                                print(msg)
                                if self.enable_logging:
                                    logging.info(msg)
                            self.no_face_time = 0
                            consecutive_failures = 0
                        else:
                            self.no_face_time += self.check_interval
                            consecutive_failures += 1
                            
                            status = "❌" if consecutive_failures > 2 else "⚠️"
                            msg = f"{status} No face detected - Elapsed: {self.no_face_time}s / {self.no_face_threshold}s"
                            if info['face_count'] > 0 and not info['is_live']:
                                msg += " (Liveness check failed - possible photo/video)"
                            print(msg)
                            
                            # Lock if threshold exceeded
                            if self.no_face_time >= self.no_face_threshold:
                                self.lock_computer()
                                self.no_face_time = 0
                                consecutive_failures = 0
                                time.sleep(5)  # Brief pause after locking
                        
                    except Exception as e:
                        logging.error(f"Detection error: {e}")
                    
                    self.last_check_time = current_time
                
                time.sleep(0.5)  # Small sleep to prevent high CPU usage
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping face detection auto-lock...")
            print(f"📊 Statistics:")
            print(f"   - Total locks triggered: {self.lock_count}")
            print(f"   - Detector used: {self.detector_type}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        self.camera.release()
        if MEDIAPIPE_AVAILABLE:
            self.face_detection.close()
        cv2.destroyAllWindows()
        logging.info("Camera released. System stopped.")
        print("✅ Camera released. Goodbye!")


if __name__ == "__main__":
    # ==================== CONFIGURATION ====================
    CHECK_INTERVAL = 2          # Check every 2 seconds (faster)
    LOCK_THRESHOLD = 10         # Lock after 10 seconds without face
    MIN_CONFIDENCE = 0.7        # Detection confidence (0.5-1.0)
    ENABLE_LIVENESS = True      # Enable basic liveness detection
    ENABLE_LOGGING = True       # Log to file
    # ========================================================
    
    print("Starting Advanced Face Detection Auto-Lock System...")
    
    if not MEDIAPIPE_AVAILABLE:
        print("\n⚠️  WARNING: MediaPipe not installed!")
        print("For best performance, install it:")
        print("   pip install mediapipe")
        print("\nUsing Haar Cascade fallback (less accurate)\n")
        time.sleep(3)
    
    try:
        detector = AdvancedFaceDetectionLock(
            check_interval=CHECK_INTERVAL,
            no_face_threshold=LOCK_THRESHOLD,
            min_detection_confidence=MIN_CONFIDENCE,
            enable_liveness=ENABLE_LIVENESS,
            enable_logging=ENABLE_LOGGING
        )
        detector.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Install required packages:")
        print("   pip install opencv-python mediapipe")
        print("2. Ensure camera is not in use by another application")
        print("3. Check camera permissions in your OS settings")
        logging.exception("Fatal error occurred")