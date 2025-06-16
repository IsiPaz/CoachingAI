import cv2
import numpy as np
import mediapipe as mp
import torch
import time
from typing import Dict, Optional, Tuple, List
from collections import deque


class IrisHandler:
    """
    Handles eye tracking, visual attention detection, and fatigue/drowsiness detection using MediaPipe.
    GPU-optimized where possible.
    """
    
    def __init__(self, 
                 device: str = "cuda:0",
                 attention_threshold: float = 0.5,
                 blink_threshold: float = 0.2,
                 fatigue_window_size: int = 150):
        """
        Initialize the Iris handler.
        
        Args:
            device: Device to run models on (cuda:0, cpu)
            attention_threshold: Threshold for determining if eyes are looking at camera
            blink_threshold: Eye aspect ratio threshold for blink detection
            fatigue_window_size: Number of frames to analyze for fatigue detection
        """
        self.device = device
        self.attention_threshold = attention_threshold
        self.blink_threshold = blink_threshold
        self.fatigue_window_size = fatigue_window_size
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Enables iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmark indices
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # Left eye landmarks
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # Right eye landmarks
        
        # Iris landmark indices (468-477 for both eyes)
        self.LEFT_IRIS_CENTER = 468
        self.RIGHT_IRIS_CENTER = 473
        
        # Blink and fatigue tracking
        self.blink_history = deque(maxlen=fatigue_window_size)
        self.eye_aspect_ratio_history = deque(maxlen=fatigue_window_size)
        self.attention_history = deque(maxlen=fatigue_window_size)
        
        # Frame counters
        self.total_blinks = 0
        self.frames_since_last_blink = 0
        self.consecutive_closed_frames = 0
        
        # GPU optimization for calculations if available
        self.use_gpu = device.startswith('cuda') and torch.cuda.is_available()
        if self.use_gpu:
            self.gpu_device = torch.device(device)
        
        print(f"IrisHandler initialized on device: {device}")
        print("Eye tracking and fatigue detection enabled")
        
    def calculate_eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection.
        
        Args:
            eye_landmarks: Array of eye landmark coordinates
            
        Returns:
            Eye aspect ratio value
        """
        # Compute euclidean distances between vertical eye landmarks
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Compute euclidean distance between horizontal eye landmarks
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # Eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
        
    def calculate_gaze_direction(self, 
                               face_landmarks: np.ndarray,
                               image_shape: Tuple[int, int]) -> Tuple[float, float, float]:
        """
        Calculate gaze direction based on iris position relative to eye corners.
        
        Args:
            face_landmarks: All face landmarks from MediaPipe
            image_shape: Shape of the input image (height, width)
            
        Returns:
            Tuple of (horizontal_gaze, vertical_gaze, gaze_magnitude)
        """
        h, w = image_shape[:2]
        
        # Get iris centers
        left_iris = face_landmarks[self.LEFT_IRIS_CENTER]
        right_iris = face_landmarks[self.RIGHT_IRIS_CENTER]
        
        # Get eye corners for reference
        left_inner = face_landmarks[133]
        left_outer = face_landmarks[33]
        right_inner = face_landmarks[362]
        right_outer = face_landmarks[263]
        
        # Calculate normalized iris positions
        left_eye_width = np.linalg.norm(left_outer - left_inner)
        right_eye_width = np.linalg.norm(right_outer - right_inner)
        
        # Calculate relative iris positions
        left_iris_relative = (left_iris - left_inner) / left_eye_width
        right_iris_relative = (right_iris - right_inner) / right_eye_width
        
        # Average both eyes for overall gaze
        avg_horizontal = (left_iris_relative[0] + right_iris_relative[0]) / 2 - 0.5
        avg_vertical = (left_iris_relative[1] + right_iris_relative[1]) / 2 - 0.5
        
        # Calculate gaze magnitude
        gaze_magnitude = np.sqrt(avg_horizontal**2 + avg_vertical**2)
        
        return avg_horizontal, avg_vertical, gaze_magnitude
        
    def detect_visual_attention(self, gaze_magnitude: float) -> bool:
        """
        Detect if the person is looking at the camera based on gaze magnitude.
        
        Args:
            gaze_magnitude: Magnitude of gaze vector
            
        Returns:
            True if looking at camera, False otherwise
        """
        return gaze_magnitude < self.attention_threshold
        
    def detect_blink(self, left_ear: float, right_ear: float) -> bool:
        """
        Detect if a blink occurred based on eye aspect ratios.
        
        Args:
            left_ear: Left eye aspect ratio
            right_ear: Right eye aspect ratio
            
        Returns:
            True if blink detected, False otherwise
        """
        avg_ear = (left_ear + right_ear) / 2.0
        return avg_ear < self.blink_threshold
        
    def calculate_fatigue_metrics(self) -> Dict[str, float]:
        """
        Calculate fatigue and drowsiness metrics based on eye behavior history.
        
        Returns:
            Dictionary containing fatigue metrics
        """
        if len(self.blink_history) < 30:  # Need minimum history
            return {
                "blink_rate": 0.0,
                "average_ear": 0.5,
                "attention_score": 1.0,
                "fatigue_level": 0.0,
                "drowsiness_alert": False
            }
            
        # Calculate blink rate (blinks per minute)
        recent_blinks = sum(self.blink_history)
        time_window = len(self.blink_history) / 30.0  # Assuming 30 FPS
        blink_rate = (recent_blinks / time_window) * 60.0
        
        # Calculate average EAR
        avg_ear = np.mean(self.eye_aspect_ratio_history) if self.eye_aspect_ratio_history else 0.5
        
        # Calculate attention score
        attention_score = sum(self.attention_history) / len(self.attention_history) if self.attention_history else 1.0
        
        # Calculate fatigue level (0-1)
        fatigue_indicators = []
        
        # High blink rate (>20 blinks/min indicates fatigue)
        if blink_rate > 20:
            fatigue_indicators.append(min((blink_rate - 20) / 20, 1.0))
            
        # Low average EAR (droopy eyes)
        if avg_ear < 0.25:
            fatigue_indicators.append(min((0.25 - avg_ear) / 0.1, 1.0))
            
        # Low attention score
        if attention_score < 0.5:
            fatigue_indicators.append(1.0 - attention_score)
            
        # Long eye closure
        if self.consecutive_closed_frames > 15:  # More than 0.5 seconds at 15 FPS
            fatigue_indicators.append(min(self.consecutive_closed_frames / 15, 1.0))
            
        fatigue_level = np.mean(fatigue_indicators) if fatigue_indicators else 0.0
        
        # Drowsiness alert if fatigue level is high
        drowsiness_alert = fatigue_level > 0.7
        
        return {
            "blink_rate": blink_rate,
            "average_ear": avg_ear,
            "attention_score": attention_score,
            "fatigue_level": fatigue_level,
            "drowsiness_alert": drowsiness_alert
        }
        
    def process_frame(self, frame_rgb: np.ndarray) -> Optional[Dict]:
        """
        Process a single frame for iris tracking and analysis.
        
        Args:
            frame_rgb: Input frame in RGB format
            
        Returns:
            Dictionary containing iris tracking results or None if no face detected
        """
        # Run MediaPipe face mesh detection
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None
            
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert landmarks to numpy array
        h, w = frame_rgb.shape[:2]
        landmarks_array = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
        
        # Extract eye landmarks
        left_eye_landmarks = landmarks_array[self.LEFT_EYE_INDICES]
        right_eye_landmarks = landmarks_array[self.RIGHT_EYE_INDICES]
        
        # Calculate eye aspect ratios
        left_ear = self.calculate_eye_aspect_ratio(left_eye_landmarks)
        right_ear = self.calculate_eye_aspect_ratio(right_eye_landmarks)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Detect blink
        is_blinking = self.detect_blink(left_ear, right_ear)
        
        # Update blink tracking
        if is_blinking:
            if self.frames_since_last_blink > 5:  # Debounce blinks
                self.total_blinks += 1
                self.blink_history.append(1)
            else:
                self.blink_history.append(0)
            self.frames_since_last_blink = 0
            self.consecutive_closed_frames += 1
        else:
            self.blink_history.append(0)
            self.frames_since_last_blink += 1
            self.consecutive_closed_frames = 0
            
        # Update EAR history
        self.eye_aspect_ratio_history.append(avg_ear)
        
        # Calculate gaze direction
        h_gaze, v_gaze, gaze_magnitude = self.calculate_gaze_direction(landmarks_array, frame_rgb.shape)
        
        # Detect visual attention
        is_looking_at_camera = self.detect_visual_attention(gaze_magnitude)
        self.attention_history.append(1 if is_looking_at_camera else 0)
        
        # Calculate fatigue metrics
        fatigue_metrics = self.calculate_fatigue_metrics()
        
        # Get iris positions for visualization
        left_iris_pos = landmarks_array[self.LEFT_IRIS_CENTER].astype(int)
        right_iris_pos = landmarks_array[self.RIGHT_IRIS_CENTER].astype(int)
        
        return {
            "left_ear": left_ear,
            "right_ear": right_ear,
            "average_ear": avg_ear,
            "is_blinking": is_blinking,
            "total_blinks": self.total_blinks,
            "horizontal_gaze": h_gaze,
            "vertical_gaze": v_gaze,
            "gaze_magnitude": gaze_magnitude,
            "is_looking_at_camera": is_looking_at_camera,
            "left_iris_position": left_iris_pos,
            "right_iris_position": right_iris_pos,
            "left_eye_landmarks": left_eye_landmarks.astype(int),
            "right_eye_landmarks": right_eye_landmarks.astype(int),
            "fatigue_metrics": fatigue_metrics
        }
        
    def draw_iris_visualization(self, 
                              frame: np.ndarray, 
                              iris_info: Dict) -> np.ndarray:
        """
        Draw iris tracking visualization on frame.
        
        Args:
            frame: Input frame
            iris_info: Iris tracking information
            
        Returns:
            Frame with visualization
        """
        vis_frame = frame.copy()
        
        # Draw eye landmarks
        for landmark in iris_info["left_eye_landmarks"]:
            cv2.circle(vis_frame, tuple(landmark), 2, (0, 255, 0), -1)
        for landmark in iris_info["right_eye_landmarks"]:
            cv2.circle(vis_frame, tuple(landmark), 2, (0, 255, 0), -1)
            
        # Draw iris centers
        cv2.circle(vis_frame, tuple(iris_info["left_iris_position"]), 4, (255, 0, 0), -1)
        cv2.circle(vis_frame, tuple(iris_info["right_iris_position"]), 4, (255, 0, 0), -1)
        
        # Draw attention indicator
        color = (0, 255, 0) if iris_info["is_looking_at_camera"] else (0, 0, 255)
        cv2.putText(vis_frame, 
                   "LOOKING" if iris_info["is_looking_at_camera"] else "NOT LOOKING",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw fatigue warning if needed
        if iris_info["fatigue_metrics"]["drowsiness_alert"]:
            cv2.putText(vis_frame, "DROWSINESS ALERT!", 
                       (frame.shape[1]//2 - 100, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
        return vis_frame