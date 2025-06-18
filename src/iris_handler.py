import cv2
import numpy as np
import mediapipe as mp
import torch
import time
from typing import Dict, Optional, Tuple, List
from collections import deque


class IrisHandler:
    """
    Handles eye tracking and iris position detection using MediaPipe.
    Returns raw values for eye aperture, iris position, and blink detection.
    GPU-optimized where possible.
    """
    
    def __init__(self, 
                 device: str = "cuda:0",
                 blink_threshold: float = 0.2,
                 history_window_size: int = 150):
        """
        Initialize the Iris handler.
        
        Args:
            device: Device to run models on (cuda:0, cpu)
            blink_threshold: Eye aspect ratio threshold for blink detection
            history_window_size: Number of frames to keep in history
        """
        self.device = device
        self.blink_threshold = blink_threshold
        self.history_window_size = history_window_size
        
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
        
        # History tracking
        self.blink_history = deque(maxlen=history_window_size)
        self.eye_aspect_ratio_history = deque(maxlen=history_window_size)
        
        # Frame counters
        self.total_blinks = 0
        self.frames_since_last_blink = 0
        self.consecutive_closed_frames = 0
        
        # GPU optimization for calculations if available
        self.use_gpu = device.startswith('cuda') and torch.cuda.is_available()
        if self.use_gpu:
            self.gpu_device = torch.device(device)
        
        print(f"IrisHandler initialized on device: {device}")
        print("Eye tracking enabled - returning raw values")
        
    def calculate_eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for eye aperture measurement.
        
        Args:
            eye_landmarks: Array of eye landmark coordinates
            
        Returns:
            Eye aspect ratio value (0 = closed, ~0.3 = normal open)
        """
        # Compute euclidean distances between vertical eye landmarks
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Compute euclidean distance between horizontal eye landmarks
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # Eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
        
    def calculate_iris_position(self, 
                               face_landmarks: np.ndarray,
                               image_shape: Tuple[int, int],
                               eyes_closed: bool = False) -> Dict[str, float]:
        """
        Calculate iris position relative to eye boundaries.
        
        Args:
            face_landmarks: All face landmarks from MediaPipe
            image_shape: Shape of the input image (height, width)
            eyes_closed: Whether the eyes are currently closed
            
        Returns:
            Dictionary with iris position metrics
        """
        # Si los ojos están cerrados, devolver valores que indiquen que no hay información válida
        if eyes_closed:
            return {
                "left_iris_horizontal_offset": 0.0,
                "left_iris_vertical_offset": 0.0,
                "left_iris_centering": 0.0,
                "right_iris_horizontal_offset": 0.0,
                "right_iris_vertical_offset": 0.0,
                "right_iris_centering": 0.0,
                "average_horizontal_offset": 0.0,
                "average_vertical_offset": 0.0,
                "average_centering": 0.0,
                "eyes_status": "closed"
            }
        
        h, w = image_shape[:2]
        
        # Get iris centers
        left_iris = face_landmarks[self.LEFT_IRIS_CENTER]
        right_iris = face_landmarks[self.RIGHT_IRIS_CENTER]
        
        # Get eye corners for reference
        left_inner = face_landmarks[133]
        left_outer = face_landmarks[33]
        right_inner = face_landmarks[362]
        right_outer = face_landmarks[263]
        
        # Get eye top and bottom for vertical reference
        left_top = face_landmarks[159]
        left_bottom = face_landmarks[145]
        right_top = face_landmarks[386]
        right_bottom = face_landmarks[374]
        
        # Calculate eye dimensions
        left_eye_width = np.linalg.norm(left_outer - left_inner)
        right_eye_width = np.linalg.norm(right_outer - right_inner)
        left_eye_height = np.linalg.norm(left_top - left_bottom)
        right_eye_height = np.linalg.norm(right_top - right_bottom)
        
        # Calculate iris position relative to eye center (normalized -1 to 1)
        left_eye_center = (left_inner + left_outer) / 2
        right_eye_center = (right_inner + right_outer) / 2
        
        left_iris_h_offset = (left_iris[0] - left_eye_center[0]) / (left_eye_width / 2)
        left_iris_v_offset = (left_iris[1] - left_eye_center[1]) / (left_eye_height / 2)
        
        right_iris_h_offset = (right_iris[0] - right_eye_center[0]) / (right_eye_width / 2)
        right_iris_v_offset = (right_iris[1] - right_eye_center[1]) / (right_eye_height / 2)
        
        # Calculate how centered the iris is (0 = perfectly centered, 1 = at edge)
        left_iris_centering = np.sqrt(left_iris_h_offset**2 + left_iris_v_offset**2)
        right_iris_centering = np.sqrt(right_iris_h_offset**2 + right_iris_v_offset**2)
        
        return {
            "left_iris_horizontal_offset": left_iris_h_offset,
            "left_iris_vertical_offset": left_iris_v_offset,
            "left_iris_centering": left_iris_centering,
            "right_iris_horizontal_offset": right_iris_h_offset,
            "right_iris_vertical_offset": right_iris_v_offset,
            "right_iris_centering": right_iris_centering,
            "average_horizontal_offset": (left_iris_h_offset + right_iris_h_offset) / 2,
            "average_vertical_offset": (left_iris_v_offset + right_iris_v_offset) / 2,
            "average_centering": (left_iris_centering + right_iris_centering) / 2,
            "eyes_status": "open"
        }
        
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
        
    def are_eyes_closed(self, left_ear: float, right_ear: float) -> bool:
        """
        Determine if eyes are currently closed (not just blinking).
        
        Args:
            left_ear: Left eye aspect ratio
            right_ear: Right eye aspect ratio
            
        Returns:
            True if eyes are closed, False otherwise
        """
        # Usar un threshold ligeramente más bajo para detectar ojos cerrados
        # vs parpadeos rápidos
        closed_threshold = self.blink_threshold * 0.8
        avg_ear = (left_ear + right_ear) / 2.0
        return avg_ear < closed_threshold
        
    def calculate_eye_metrics(self) -> Dict[str, float]:
        """
        Calculate various eye metrics based on history.
        
        Returns:
            Dictionary containing eye metrics
        """
        if len(self.blink_history) < 30:  # Need minimum history
            return {
                "blink_count_last_5_sec": 0,
                "average_ear_last_5_sec": 0.3,
                "eye_closure_percentage": 0.0,
                "longest_closure_frames": 0
            }
            
        # Calculate metrics over last 5 seconds (assuming 30 FPS)
        frames_5_sec = min(150, len(self.blink_history))
        recent_blinks = sum(list(self.blink_history)[-frames_5_sec:])
        
        # Average EAR over recent history
        recent_ears = list(self.eye_aspect_ratio_history)[-frames_5_sec:]
        avg_ear = np.mean(recent_ears) if recent_ears else 0.3
        
        # Percentage of time eyes were closed
        closed_frames = sum(1 for ear in recent_ears if ear < self.blink_threshold)
        closure_percentage = (closed_frames / len(recent_ears) * 100) if recent_ears else 0.0
        
        return {
            "blink_count_last_5_sec": recent_blinks,
            "average_ear_last_5_sec": avg_ear,
            "eye_closure_percentage": closure_percentage,
            "longest_closure_frames": self.consecutive_closed_frames
        }
        
    def process_frame(self, frame_rgb: np.ndarray) -> Optional[Dict]:
        """
        Process a single frame for iris tracking and analysis.
        
        Args:
            frame_rgb: Input frame in RGB format
            
        Returns:
            Dictionary containing raw iris tracking values or None if no face detected
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
        
        # Calculate eye aspect ratios (eye aperture)
        left_ear = self.calculate_eye_aspect_ratio(left_eye_landmarks)
        right_ear = self.calculate_eye_aspect_ratio(right_eye_landmarks)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Detect blink and check if eyes are closed
        is_blinking = self.detect_blink(left_ear, right_ear)
        eyes_closed = self.are_eyes_closed(left_ear, right_ear)
        
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
        
        # Calculate iris position (pasando el estado de ojos cerrados)
        iris_position = self.calculate_iris_position(landmarks_array, frame_rgb.shape, eyes_closed)
        
        # Calculate eye metrics
        eye_metrics = self.calculate_eye_metrics()
        
        # Get iris positions for visualization (solo si los ojos están abiertos)
        left_iris_pos = None
        right_iris_pos = None
        if not eyes_closed:
            left_iris_pos = landmarks_array[self.LEFT_IRIS_CENTER].astype(int)
            right_iris_pos = landmarks_array[self.RIGHT_IRIS_CENTER].astype(int)
        
        return {
            # Raw eye aperture values
            "left_eye_aperture": left_ear,
            "right_eye_aperture": right_ear,
            "average_eye_aperture": avg_ear,
            
            # Blink information
            "is_blinking": is_blinking,
            "eyes_closed": eyes_closed,  # Nueva información
            "total_blinks": self.total_blinks,
            "frames_since_last_blink": self.frames_since_last_blink,
            
            # Iris position data
            "iris_position": iris_position,
            
            # Eye metrics
            "eye_metrics": eye_metrics,
            
            # Visualization data (None si los ojos están cerrados)
            "left_iris_position": left_iris_pos,
            "right_iris_position": right_iris_pos,
            "left_eye_landmarks": left_eye_landmarks.astype(int),
            "right_eye_landmarks": right_eye_landmarks.astype(int)
        }
        
    def draw_iris_visualization(self, 
                              frame: np.ndarray, 
                              iris_info: Dict,
                              debug: bool = False) -> np.ndarray:
        """
        Draw iris tracking visualization on frame.
        
        Args:
            frame: Input frame
            iris_info: Iris tracking information
            debug: Whether to show debug visualization
            
        Returns:
            Frame with visualization
        """
        vis_frame = frame.copy()
        
        # Solo dibujar si debug está activado
        if debug:
            # Draw eye landmarks
            for landmark in iris_info["left_eye_landmarks"]:
                cv2.circle(vis_frame, tuple(landmark), 2, (0, 255, 0), -1)
            for landmark in iris_info["right_eye_landmarks"]:
                cv2.circle(vis_frame, tuple(landmark), 2, (0, 255, 0), -1)
            
            # Solo dibujar iris centers si los ojos están abiertos
            if not iris_info["eyes_closed"]:
                if iris_info["left_iris_position"] is not None:
                    cv2.circle(vis_frame, tuple(iris_info["left_iris_position"]), 4, (255, 0, 0), -1)
                if iris_info["right_iris_position"] is not None:
                    cv2.circle(vis_frame, tuple(iris_info["right_iris_position"]), 4, (255, 0, 0), -1)
            
            # Draw eye aperture values y estado de los ojos
            aperture_text = f"L: {iris_info['left_eye_aperture']:.3f} R: {iris_info['right_eye_aperture']:.3f}"
            status_text = f"Eyes: {'CLOSED' if iris_info['eyes_closed'] else 'OPEN'}"
            
            cv2.putText(vis_frame, aperture_text,
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_frame, status_text,
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 0, 255) if iris_info['eyes_closed'] else (0, 255, 0), 2)
            
        return vis_frame