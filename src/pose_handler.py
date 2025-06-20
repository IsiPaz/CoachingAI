import cv2
import numpy as np
import mediapipe as mp
import torch
import math
from typing import Dict, Optional, Tuple, List
from collections import deque


class PoseHandler:
    """
    Handles body pose tracking and posture analysis using MediaPipe Pose.
    Returns comprehensive posture metrics including trunk inclination, 
    shoulder symmetry, and head orientation.
    GPU-optimized where possible.
    """
    
    def __init__(self, 
                 device: str = "cuda:0",
                 history_window_size: int = 30):
        """
        Initialize the Pose handler.
        
        Args:
            device: Device to run models on (cuda:0, cpu)
            history_window_size: Number of frames to keep in history for smoothing
        """
        self.device = device
        self.history_window_size = history_window_size
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Balance between accuracy and speed
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Key landmark indices for pose analysis
        self.POSE_LANDMARKS = {
            'nose': 0,
            'left_ear': 7, 'right_ear': 8,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_pinky': 17, 'right_pinky': 18,
            'left_index': 19, 'right_index': 20,
            'left_thumb': 21, 'right_thumb': 22,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_heel': 29, 'right_heel': 30,
            'left_foot_index': 31, 'right_foot_index': 32
        }
        
        # History tracking for smoothing
        self.posture_history = deque(maxlen=history_window_size)
        
        # GPU optimization for calculations if available
        self.use_gpu = device.startswith('cuda') and torch.cuda.is_available()
        if self.use_gpu:
            self.gpu_device = torch.device(device)
        
        print(f"PoseHandler initialized on device: {device}")
        print("Pose tracking enabled - analyzing posture metrics")
        
    def calculate_angle(self, point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
        """
        Calculate angle between three points in degrees.
        
        Args:
            point1: First point (x, y)
            point2: Middle point (vertex of angle)
            point3: Third point (x, y)
            
        Returns:
            Angle in degrees
        """
        # Vector from point2 to point1
        v1 = point1 - point2
        # Vector from point2 to point3
        v2 = point3 - point2
        
        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Prevent numerical errors
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def calculate_trunk_inclination(self, landmarks: np.ndarray) -> Tuple[float, float]:
        """
        Calculate trunk inclination in forward/backward and left/right directions.
        
        Args:
            landmarks: Pose landmarks array
            
        Returns:
            Tuple of (forward_backward_angle, left_right_angle) in degrees
        """
        # Get shoulder and hip midpoints
        left_shoulder = landmarks[self.POSE_LANDMARKS['left_shoulder']]
        right_shoulder = landmarks[self.POSE_LANDMARKS['right_shoulder']]
        left_hip = landmarks[self.POSE_LANDMARKS['left_hip']]
        right_hip = landmarks[self.POSE_LANDMARKS['right_hip']]
        
        shoulder_midpoint = (left_shoulder + right_shoulder) / 2
        hip_midpoint = (left_hip + right_hip) / 2
        
        # Calculate trunk vector
        trunk_vector = shoulder_midpoint - hip_midpoint
        
        # Forward/backward inclination (using y and z if available, else approximate)
        # Positive = forward lean, Negative = backward lean
        fb_angle = np.degrees(np.arctan2(trunk_vector[0], abs(trunk_vector[1])))
        
        # Left/right inclination
        # Positive = right lean, Negative = left lean
        lr_angle = np.degrees(np.arctan2(trunk_vector[0], trunk_vector[1]))
        
        return fb_angle, lr_angle
    
    def calculate_shoulder_symmetry(self, landmarks: np.ndarray) -> Tuple[float, float]:
        """
        Calculate shoulder symmetry and asymmetry with more realistic thresholds.
        
        Args:
            landmarks: Pose landmarks array
            
        Returns:
            Tuple of (asymmetry_angle, symmetry_score)
        """
        left_shoulder = landmarks[self.POSE_LANDMARKS['left_shoulder']]
        right_shoulder = landmarks[self.POSE_LANDMARKS['right_shoulder']]
        
        # Calculate height difference
        height_diff = right_shoulder[1] - left_shoulder[1]
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        
        # Convert to angle
        asymmetry_angle = np.degrees(np.arctan2(height_diff, shoulder_width))
        
        # More realistic symmetry score - up to 8° difference is still good
        max_reasonable_angle = 20.0  # Increased from 15.0
        good_threshold = 8.0  # Perfect range
        
        if abs(asymmetry_angle) <= good_threshold:
            symmetry_score = 1.0
        else:
            symmetry_score = max(0.3, 1.0 - abs(asymmetry_angle) / max_reasonable_angle)
        
        return asymmetry_angle, symmetry_score
    
    def calculate_head_orientation(self, landmarks: np.ndarray) -> Tuple[float, float]:
        """
        Calculate head tilt and turn angles.
        
        Args:
            landmarks: Pose landmarks array
            
        Returns:
            Tuple of (head_tilt, head_turn) in degrees
        """
        # Get facial landmarks
        left_ear = landmarks[self.POSE_LANDMARKS['left_ear']]
        right_ear = landmarks[self.POSE_LANDMARKS['right_ear']]
        nose = landmarks[self.POSE_LANDMARKS['nose']]
        
        # Head tilt (roll) - ear to ear line relative to horizontal
        ear_vector = right_ear - left_ear
        head_tilt = np.degrees(np.arctan2(ear_vector[1], ear_vector[0]))
        
        # Head turn (yaw) - nose position relative to ear midpoint
        ear_midpoint = (left_ear + right_ear) / 2
        nose_offset = nose[0] - ear_midpoint[0]
        ear_width = abs(right_ear[0] - left_ear[0])
        head_turn = np.degrees(np.arctan2(nose_offset, ear_width)) * 2  # Amplify for visibility
        
        return head_tilt, head_turn
    
    def calculate_overall_posture_score(self, posture_metrics: Dict) -> float:
        """
        Calculate overall posture score based on all metrics with flexible ranges.
        
        Args:
            posture_metrics: Dictionary containing all posture measurements
            
        Returns:
            Overall posture score (0.0 to 1.0, higher is better)
        """
        scores = []
        
        # Trunk inclination score (slightly more flexible)
        trunk_fb = abs(posture_metrics['trunk_inclination_fb'])
        trunk_lr = abs(posture_metrics['trunk_inclination_lr'])
        
        # Good range: 0-17°, Fair: 17-28°, Poor: >28°
        if trunk_fb <= 17 and trunk_lr <= 17:
            trunk_score = 1.0
        elif trunk_fb <= 28 and trunk_lr <= 28:
            trunk_score = max(0.65, 1.0 - (trunk_fb + trunk_lr) / 56.0)
        else:
            trunk_score = max(0.3, 1.0 - (trunk_fb + trunk_lr) / 85.0)
        scores.append(trunk_score)
        
        # Symmetry scores (slightly more generous)
        shoulder_score = posture_metrics['shoulder_symmetry_score']
        # Small boost for decent symmetry
        if shoulder_score > 0.8:
            shoulder_score = min(1.0, shoulder_score + 0.1)
        elif shoulder_score > 0.65:
            shoulder_score = min(1.0, shoulder_score + 0.05)
        scores.append(shoulder_score)
        
        # Head orientation score (slightly more permissive)
        head_tilt = abs(posture_metrics['head_tilt'])
        head_turn = abs(posture_metrics['head_turn'])
        
        # Good range: 0-20°, Fair: 20-38°, Poor: >38°
        if head_tilt <= 20 and head_turn <= 20:
            head_score = 1.0
        elif head_tilt <= 38 and head_turn <= 38:
            head_score = max(0.6, 1.0 - (head_tilt + head_turn) / 76.0)
        else:
            head_score = max(0.2, 1.0 - (head_tilt + head_turn) / 110.0)
        scores.append(head_score)
        
        # Calculate weighted average with small boost
        weights = [0.4, 0.35, 0.25]  # trunk, shoulder, head
        weighted_score = sum(score * weight for score, weight in zip(scores, weights))
        
        # Apply a small boost if most components are decent
        if sum(1 for score in scores if score > 0.6) >= 2:
            weighted_score = min(1.0, weighted_score + 0.07)
        
        return weighted_score
    
    def process_frame(self, frame_rgb: np.ndarray) -> Optional[Dict]:
        """
        Process a single frame for pose tracking and posture analysis.
        
        Args:
            frame_rgb: Input frame in RGB format
            
        Returns:
            Dictionary containing comprehensive posture metrics or None if no pose detected
        """
        # Run MediaPipe pose detection
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None
        
        # Convert landmarks to numpy array
        h, w = frame_rgb.shape[:2]
        landmarks_array = np.array([(lm.x * w, lm.y * h) for lm in results.pose_landmarks.landmark])
        
        # Calculate all posture metrics
        trunk_fb, trunk_lr = self.calculate_trunk_inclination(landmarks_array)
        shoulder_asym, shoulder_sym = self.calculate_shoulder_symmetry(landmarks_array)
        head_tilt, head_turn = self.calculate_head_orientation(landmarks_array)
        
        # Create posture metrics dictionary
        posture_metrics = {
            'trunk_inclination_fb': trunk_fb,
            'trunk_inclination_lr': trunk_lr,
            'shoulder_asymmetry': shoulder_asym,
            'shoulder_symmetry_score': shoulder_sym,
            'head_tilt': head_tilt,
            'head_turn': head_turn,
        }
        
        # Calculate overall posture score
        posture_metrics['overall_posture_score'] = self.calculate_overall_posture_score(posture_metrics)
        
        # Add to history for smoothing
        self.posture_history.append(posture_metrics)
        
        # Calculate smoothed metrics if we have enough history
        smoothed_metrics = posture_metrics.copy()
        if len(self.posture_history) >= 5:
            # Smooth numerical values
            numeric_keys = ['trunk_inclination_fb', 'trunk_inclination_lr', 
                          'shoulder_asymmetry', 'head_tilt', 'head_turn']
            
            for key in numeric_keys:
                values = [metrics[key] for metrics in list(self.posture_history)[-5:]]
                smoothed_metrics[key] = np.mean(values)
        
        return {
            'posture_metrics': smoothed_metrics,
            'raw_landmarks': landmarks_array,
            'pose_landmarks': results.pose_landmarks,
            'pose_detected': True
        }
    
    def draw_pose_visualization(self, 
                              frame: np.ndarray, 
                              pose_info: Dict,
                              debug: bool = False) -> np.ndarray:
        """
        Draw pose tracking visualization on frame.
        
        Args:
            frame: Input frame
            pose_info: Pose tracking information
            debug: Whether to show debug visualization
            
        Returns:
            Frame with visualization
        """
        vis_frame = frame.copy()
        
        # Only draw if debug is enabled
        if debug and pose_info['pose_detected']:
            # Draw ONLY body pose landmarks and connections (exclude face)
            # Create custom connections excluding facial connections
            body_connections = [
                # Torso
                (11, 12), (11, 23), (12, 24), (23, 24),
                # Left arm
                (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
                (17, 19), (19, 21),
                # Right arm  
                (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                (18, 20), (20, 22),
                # Left leg
                (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
                # Right leg
                (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
            ]
            
            # Draw only body landmarks (exclude face landmarks 0-10)
            landmarks = pose_info['pose_landmarks'].landmark
            for i in range(11, len(landmarks)):  # Start from shoulder landmarks
                if landmarks[i].visibility > 0.5:
                    x = int(landmarks[i].x * vis_frame.shape[1])
                    y = int(landmarks[i].y * vis_frame.shape[0])
                    cv2.circle(vis_frame, (x, y), 3, (0, 255, 0), -1)
            
            # Draw body connections
            for connection in body_connections:
                start_idx, end_idx = connection
                start_landmark = landmarks[start_idx]
                end_landmark = landmarks[end_idx]
                
                if start_landmark.visibility > 0.5 and end_landmark.visibility > 0.5:
                    start_x = int(start_landmark.x * vis_frame.shape[1])
                    start_y = int(start_landmark.y * vis_frame.shape[0])
                    end_x = int(end_landmark.x * vis_frame.shape[1])
                    end_y = int(end_landmark.y * vis_frame.shape[0])
                    
                    cv2.line(vis_frame, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
            
            # Draw key posture information on frame
            posture = pose_info['posture_metrics']
            
            # Create status text
            status_lines = [
                f"Trunk: FB:{posture['trunk_inclination_fb']:+.1f}° LR:{posture['trunk_inclination_lr']:+.1f}°",
                f"Shoulders: {posture['shoulder_asymmetry']:+.1f}° (Sym:{posture['shoulder_symmetry_score']:.2f})",
                f"Head: Tilt:{posture['head_tilt']:+.1f}° Turn:{posture['head_turn']:+.1f}°",
                f"Posture Score: {posture['overall_posture_score']:.2f}"
            ]
            
            # Draw status text
            y_offset = 140
            for i, line in enumerate(status_lines):
                cv2.putText(vis_frame, line, (10, y_offset + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw posture quality indicator with slightly more permissive thresholds
            score = posture['overall_posture_score']
            if score > 0.68:  # Slightly lower than before
                color = (0, 255, 0)
                quality_text = "Excellent"
            elif score > 0.48:  # Slightly lower than before
                color = (0, 255, 255)
                quality_text = "Good"
            else:
                color = (0, 0, 255)
                quality_text = "Needs Improvement"
                
            cv2.putText(vis_frame, f"Posture: {quality_text}", 
                       (10, y_offset + len(status_lines) * 20 + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return vis_frame