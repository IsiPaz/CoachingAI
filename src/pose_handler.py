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
    shoulder/hip symmetry, head orientation, and sitting posture analysis.
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
        Calculate shoulder symmetry and asymmetry.
        
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
        
        # Symmetry score (1.0 = perfect symmetry, 0.0 = maximum asymmetry)
        max_reasonable_angle = 15.0  # degrees
        symmetry_score = max(0.0, 1.0 - abs(asymmetry_angle) / max_reasonable_angle)
        
        return asymmetry_angle, symmetry_score
    
    def calculate_hip_symmetry(self, landmarks: np.ndarray) -> Tuple[float, float]:
        """
        Calculate hip symmetry and asymmetry.
        
        Args:
            landmarks: Pose landmarks array
            
        Returns:
            Tuple of (asymmetry_angle, symmetry_score)
        """
        left_hip = landmarks[self.POSE_LANDMARKS['left_hip']]
        right_hip = landmarks[self.POSE_LANDMARKS['right_hip']]
        
        # Calculate height difference
        height_diff = right_hip[1] - left_hip[1]
        hip_width = abs(right_hip[0] - left_hip[0])
        
        # Convert to angle
        asymmetry_angle = np.degrees(np.arctan2(height_diff, hip_width))
        
        # Symmetry score
        max_reasonable_angle = 10.0  # degrees
        symmetry_score = max(0.0, 1.0 - abs(asymmetry_angle) / max_reasonable_angle)
        
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
    
    def detect_sitting_posture(self, landmarks: np.ndarray) -> Tuple[bool, float, str]:
        """
        Detect if person is sitting and analyze sitting posture quality.
        
        Args:
            landmarks: Pose landmarks array
            
        Returns:
            Tuple of (is_sitting, back_angle, posture_quality)
        """
        # Get key points for sitting detection
        left_hip = landmarks[self.POSE_LANDMARKS['left_hip']]
        right_hip = landmarks[self.POSE_LANDMARKS['right_hip']]
        left_knee = landmarks[self.POSE_LANDMARKS['left_knee']]
        right_knee = landmarks[self.POSE_LANDMARKS['right_knee']]
        left_shoulder = landmarks[self.POSE_LANDMARKS['left_shoulder']]
        right_shoulder = landmarks[self.POSE_LANDMARKS['right_shoulder']]
        
        # Check if sitting based on hip-knee relationship
        hip_midpoint = (left_hip + right_hip) / 2
        knee_midpoint = (left_knee + right_knee) / 2
        
        # In sitting position, knees are typically at similar height or slightly lower than hips
        hip_knee_diff = hip_midpoint[1] - knee_midpoint[1]
        is_sitting = -50 < hip_knee_diff < 100  # Threshold for sitting detection
        
        if is_sitting:
            # Calculate back angle for sitting posture
            shoulder_midpoint = (left_shoulder + right_shoulder) / 2
            back_vector = shoulder_midpoint - hip_midpoint
            back_angle = 90 + np.degrees(np.arctan2(back_vector[1], abs(back_vector[0])))
            
            # Determine posture quality
            if 75 <= back_angle <= 105:
                quality = "Good"
            elif 60 <= back_angle < 75 or 105 < back_angle <= 120:
                quality = "Fair"
            else:
                quality = "Poor"
                
            return True, back_angle, quality
        else:
            return False, 0.0, "Standing"
    
    def calculate_overall_posture_score(self, posture_metrics: Dict) -> float:
        """
        Calculate overall posture score based on all metrics.
        
        Args:
            posture_metrics: Dictionary containing all posture measurements
            
        Returns:
            Overall posture score (0.0 to 1.0, higher is better)
        """
        scores = []
        
        # Trunk inclination score
        trunk_fb = abs(posture_metrics['trunk_inclination_fb'])
        trunk_lr = abs(posture_metrics['trunk_inclination_lr'])
        trunk_score = max(0.0, 1.0 - (trunk_fb + trunk_lr) / 30.0)
        scores.append(trunk_score)
        
        # Symmetry scores
        scores.append(posture_metrics['shoulder_symmetry_score'])
        scores.append(posture_metrics['hip_symmetry_score'])
        
        # Head orientation score
        head_tilt = abs(posture_metrics['head_tilt'])
        head_turn = abs(posture_metrics['head_turn'])
        head_score = max(0.0, 1.0 - (head_tilt + head_turn) / 40.0)
        scores.append(head_score)
        
        return np.mean(scores)
    
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
        hip_asym, hip_sym = self.calculate_hip_symmetry(landmarks_array)
        head_tilt, head_turn = self.calculate_head_orientation(landmarks_array)
        is_sitting, back_angle, posture_quality = self.detect_sitting_posture(landmarks_array)
        
        # Create posture metrics dictionary
        posture_metrics = {
            'trunk_inclination_fb': trunk_fb,
            'trunk_inclination_lr': trunk_lr,
            'shoulder_asymmetry': shoulder_asym,
            'shoulder_symmetry_score': shoulder_sym,
            'hip_asymmetry': hip_asym,
            'hip_symmetry_score': hip_sym,
            'head_tilt': head_tilt,
            'head_turn': head_turn,
            'is_sitting': is_sitting,
            'sitting_back_angle': back_angle,
            'sitting_posture_quality': posture_quality,
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
                          'shoulder_asymmetry', 'hip_asymmetry', 
                          'head_tilt', 'head_turn', 'sitting_back_angle']
            
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
            
            if posture['is_sitting']:
                status_lines.append(f"Sitting: {posture['sitting_back_angle']:.1f}° ({posture['sitting_posture_quality']})")
            else:
                status_lines.append("Standing")
            
            # Draw status text
            y_offset = 140
            for i, line in enumerate(status_lines):
                cv2.putText(vis_frame, line, (10, y_offset + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw posture quality indicator
            score = posture['overall_posture_score']
            color = (0, 255, 0) if score > 0.8 else (0, 255, 255) if score > 0.6 else (0, 0, 255)
            cv2.putText(vis_frame, f"Posture: {'Excellent' if score > 0.8 else 'Good' if score > 0.6 else 'Needs Improvement'}", 
                       (10, y_offset + len(status_lines) * 20 + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return vis_frame