import cv2
import numpy as np
import torch
from typing import Dict, Optional, Tuple
from collections import deque


class DistanceHandler:
    """
    Handles camera distance estimation using face size and interpupillary distance.
    GPU-optimized for minimal performance impact.
    """
    
    def __init__(self, 
                 device: str = "cuda:0",
                 history_window_size: int = 15):
        """
        Initialize the Distance handler.
        
        Args:
            device: Device to run calculations on (cuda:0, cpu)
            history_window_size: Number of frames for smoothing
        """
        self.device = device
        self.history_window_size = history_window_size
        
        # Average interpupillary distance in mm
        self.AVG_IPD_MM = 63.0  # Average adult IPD
        
        # Optimal distance thresholds (in cm)
        self.OPTIMAL_MIN_DISTANCE = 45  # 45cm
        self.OPTIMAL_MAX_DISTANCE = 70  # 70cm
        self.TOO_CLOSE_THRESHOLD = 30   # 30cm
        self.TOO_FAR_THRESHOLD = 120    # 120cm
        
        # History for smoothing
        self.distance_history = deque(maxlen=history_window_size)
        self.face_width_history = deque(maxlen=history_window_size)
        
        # GPU optimization
        self.use_gpu = device.startswith('cuda') and torch.cuda.is_available()
        if self.use_gpu:
            self.gpu_device = torch.device(device)
        
        # Camera calibration (approximate focal length)
        self.focal_length_pixel = None
        self.calibrated = False
        
        print(f"DistanceHandler initialized on device: {device}")
        
    def estimate_focal_length(self, face_width_pixels: float, assumed_distance_cm: float = 60) -> float:
        """
        Estimate camera focal length using face width.
        
        Args:
            face_width_pixels: Face width in pixels
            assumed_distance_cm: Assumed distance for calibration
            
        Returns:
            Estimated focal length in pixels
        """
        # Average face width in cm (approximate)
        avg_face_width_cm = 14.0
        
        # focal_length = (pixel_width * distance) / real_width
        focal_length = (face_width_pixels * assumed_distance_cm) / avg_face_width_cm
        return focal_length
    
    def calculate_distance_from_face(self, face_bbox: np.ndarray, frame_shape: Tuple[int, int]) -> float:
        """
        Calculate distance using face bounding box size.
        
        Args:
            face_bbox: Face bounding box [x1, y1, x2, y2]
            frame_shape: Frame dimensions (height, width)
            
        Returns:
            Estimated distance in cm
        """
        face_width_pixels = face_bbox[2] - face_bbox[0]
        
        # Auto-calibrate on first detection
        if not self.calibrated and face_width_pixels > 50:
            self.focal_length_pixel = self.estimate_focal_length(face_width_pixels)
            self.calibrated = True
        
        if self.focal_length_pixel is None:
            # Use default focal length based on typical webcam
            self.focal_length_pixel = frame_shape[1] * 1.2
        
        # Calculate distance
        avg_face_width_cm = 14.0
        distance_cm = (avg_face_width_cm * self.focal_length_pixel) / face_width_pixels
        
        return distance_cm
    
    def calculate_distance_from_iris(self, 
                                   iris_info: Dict, 
                                   face_bbox: np.ndarray) -> Optional[float]:
        """
        Calculate distance using interpupillary distance if both eyes detected.
        
        Args:
            iris_info: Iris tracking information
            face_bbox: Face bounding box
            
        Returns:
            Estimated distance in cm or None
        """
        if iris_info is None or 'eye_coordinates' not in iris_info:
            return None
            
        eye_coords = iris_info.get('eye_coordinates', {})
        
        # Check if both eyes are detected
        if 'left_eye_center' not in eye_coords or 'right_eye_center' not in eye_coords:
            return None
            
        left_eye = eye_coords['left_eye_center']
        right_eye = eye_coords['right_eye_center']
        
        # Calculate IPD in pixels
        ipd_pixels = np.sqrt((right_eye[0] - left_eye[0])**2 + 
                            (right_eye[1] - left_eye[1])**2)
        
        if ipd_pixels < 10:  # Too small, likely error
            return None
        
        # Estimate distance using IPD
        # Using similar triangles: distance = (real_ipd * focal_length) / pixel_ipd
        if self.focal_length_pixel:
            distance_cm = (self.AVG_IPD_MM * self.focal_length_pixel) / (ipd_pixels * 10)
            return distance_cm
            
        return None
    
    def get_distance_status(self, distance_cm: float) -> Dict:
        """
        Determine distance status and recommendations.
        
        Args:
            distance_cm: Distance in centimeters
            
        Returns:
            Dictionary with status and recommendations
        """
        if distance_cm < self.TOO_CLOSE_THRESHOLD:
            status = "Too close"
            recommendation = "Move back from camera"
            quality = "poor"
            color = (0, 0, 255)  # Red
        elif distance_cm > self.TOO_FAR_THRESHOLD:
            status = "Too far"
            recommendation = "Move closer to camera"
            quality = "poor"
            color = (0, 0, 255)  # Red
        elif self.OPTIMAL_MIN_DISTANCE <= distance_cm <= self.OPTIMAL_MAX_DISTANCE:
            status = "OPTIMAL"
            recommendation = "Good distance"
            quality = "excellent"
            color = (0, 255, 0)  # Green
        else:
            # Combine all other cases into poor quality (too close or too far)
            if distance_cm < self.OPTIMAL_MIN_DISTANCE:
                status = "Too close"
                recommendation = "Move back from camera"
            else:
                status = "Too far"
                recommendation = "Move closer to camera"
            quality = "poor"
            color = (0, 0, 255)  # Red
        
        return {
            'status': status,
            'recommendation': recommendation,
            'quality': quality,
            'color': color
        }

    
    def process_frame(self, 
                     face_bbox: Optional[np.ndarray],
                     iris_info: Optional[Dict],
                     frame_shape: Tuple[int, int]) -> Optional[Dict]:
        """
        Process frame to estimate camera distance.
        
        Args:
            face_bbox: Face bounding box
            iris_info: Iris tracking information
            frame_shape: Frame dimensions
            
        Returns:
            Distance metrics dictionary or None
        """
        if face_bbox is None:
            return None
        
        # Calculate distance using face size
        face_distance = self.calculate_distance_from_face(face_bbox, frame_shape)
        
        # Try to get more accurate distance from iris if available
        iris_distance = self.calculate_distance_from_iris(iris_info, face_bbox)
        
        # Use iris distance if available (more accurate), otherwise face distance
        if iris_distance is not None and 20 < iris_distance < 200:
            primary_distance = iris_distance
            method = "iris"
        else:
            primary_distance = face_distance
            method = "face"
        
        # Add to history for smoothing
        self.distance_history.append(primary_distance)
        
        # Calculate smoothed distance
        if len(self.distance_history) >= 3:
            smoothed_distance = np.median(list(self.distance_history)[-5:])
        else:
            smoothed_distance = primary_distance
        
        # Get status
        status_info = self.get_distance_status(smoothed_distance)
        
        # Calculate face size percentage of frame
        face_width = face_bbox[2] - face_bbox[0]
        face_height = face_bbox[3] - face_bbox[1]
        face_area = face_width * face_height
        frame_area = frame_shape[0] * frame_shape[1]
        face_percentage = (face_area / frame_area) * 100
        
        return {
            'distance_cm': smoothed_distance,
            'raw_distance_cm': primary_distance,
            'distance_status': status_info['status'],
            'distance_quality': status_info['quality'],
            'recommendation': status_info['recommendation'],
            'status_color': status_info['color'],
            'measurement_method': method,
            'face_size_percentage': face_percentage,
            'optimal_range': f"{self.OPTIMAL_MIN_DISTANCE}-{self.OPTIMAL_MAX_DISTANCE}cm",
            'is_calibrated': self.calibrated
        }