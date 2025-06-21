import cv2
import numpy as np
import mediapipe as mp
import torch
import math
from typing import Dict, Optional, Tuple, List
from collections import deque


class HandHandler:
    """
    Handles hand tracking and gesture analysis using MediaPipe Hands.
    Returns comprehensive hand metrics including finger states, gestures,
    hand openness, movement patterns, and non-verbal communication indicators.
    GPU-optimized where possible.
    """
    
    def __init__(self, 
                 device: str = "cuda:0",
                 max_num_hands: int = 2,
                 history_window_size: int = 30):
        """
        Initialize the Hand handler.
        
        Args:
            device: Device to run models on (cuda:0, cpu)
            max_num_hands: Maximum number of hands to detect
            history_window_size: Number of frames to keep in history for smoothing
        """
        self.device = device
        self.max_num_hands = max_num_hands
        self.history_window_size = history_window_size
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,  # Balance between accuracy and speed
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Hand landmark indices
        self.HAND_LANDMARKS = {
            'wrist': 0,
            'thumb_cmc': 1, 'thumb_mcp': 2, 'thumb_ip': 3, 'thumb_tip': 4,
            'index_mcp': 5, 'index_pip': 6, 'index_dip': 7, 'index_tip': 8,
            'middle_mcp': 9, 'middle_pip': 10, 'middle_dip': 11, 'middle_tip': 12,
            'ring_mcp': 13, 'ring_pip': 14, 'ring_dip': 15, 'ring_tip': 16,
            'pinky_mcp': 17, 'pinky_pip': 18, 'pinky_dip': 19, 'pinky_tip': 20
        }
        
        # Finger groups for analysis
        self.FINGER_TIPS = [4, 8, 12, 16, 20]  # thumb to pinky tips
        self.FINGER_BASES = [2, 5, 9, 13, 17]  # thumb to pinky MCPs
        
        # History tracking for gesture recognition and smoothing
        self.hand_history = {
            'left': deque(maxlen=history_window_size),
            'right': deque(maxlen=history_window_size)
        }
        
        # Gesture state tracking
        self.gesture_state = {
            'left': {'prev_gesture': None, 'gesture_frames': 0},
            'right': {'prev_gesture': None, 'gesture_frames': 0}
        }
        
        # Movement tracking
        self.prev_hand_centers = {'left': None, 'right': None}
        self.movement_history = {
            'left': deque(maxlen=15),
            'right': deque(maxlen=15)
        }
        
        # GPU optimization for calculations if available
        self.use_gpu = device.startswith('cuda') and torch.cuda.is_available()
        if self.use_gpu:
            self.gpu_device = torch.device(device)
        
        print(f"HandHandler initialized on device: {device}")
        print(f"Hand tracking enabled - analyzing up to {max_num_hands} hands")
        
    def calculate_finger_curl(self, landmarks: np.ndarray, finger_idx: int) -> float:
        """
        Calculate how much a finger is curled (0 = straight, 1 = fully curled).
        
        Args:
            landmarks: Hand landmarks array
            finger_idx: Finger index (0=thumb, 1=index, 2=middle, 3=ring, 4=pinky)
            
        Returns:
            Curl value between 0 and 1
        """
        if finger_idx == 0:  # Thumb
            # For thumb, use different logic due to its unique structure
            base = landmarks[1]  # CMC
            mid = landmarks[2]   # MCP
            tip = landmarks[4]   # Tip
            
            # Calculate angle at MCP joint
            v1 = base - mid
            v2 = tip - mid
        else:
            # For other fingers
            base_idx = 5 + (finger_idx - 1) * 4  # MCP
            pip_idx = base_idx + 1
            dip_idx = base_idx + 2
            tip_idx = base_idx + 3
            
            # Calculate angle at PIP joint
            base = landmarks[base_idx]
            mid = landmarks[pip_idx]
            tip = landmarks[dip_idx]
            
            v1 = base - mid
            v2 = tip - mid
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        
        # Normalize to 0-1 range (0 = straight, 1 = curled)
        curl = angle_rad / np.pi
        return curl
    
    def calculate_hand_openness(self, landmarks: np.ndarray) -> float:
        """
        Calculate overall hand openness (0 = closed fist, 1 = open palm).
        
        Args:
            landmarks: Hand landmarks array
            
        Returns:
            Openness value between 0 and 1
        """
        # Calculate average finger curl
        finger_curls = []
        for i in range(5):
            curl = self.calculate_finger_curl(landmarks, i)
            finger_curls.append(curl)
        
        # Invert to get openness (1 - curl)
        avg_openness = 1.0 - np.mean(finger_curls)
        
        # Also consider finger spread
        spread = self.calculate_finger_spread(landmarks)
        
        # Combine openness and spread
        combined_openness = avg_openness * 0.7 + spread * 0.3
        
        return combined_openness
    
    def calculate_finger_spread(self, landmarks: np.ndarray) -> float:
        """
        Calculate how spread apart the fingers are (0 = together, 1 = max spread).
        
        Args:
            landmarks: Hand landmarks array
            
        Returns:
            Spread value between 0 and 1
        """
        # Calculate angles between adjacent fingers
        finger_tips = [landmarks[i] for i in self.FINGER_TIPS[1:]]  # Exclude thumb
        
        angles = []
        for i in range(len(finger_tips) - 1):
            v1 = finger_tips[i] - landmarks[0]  # Vector from wrist to finger
            v2 = finger_tips[i + 1] - landmarks[0]
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            angles.append(angle)
        
        # Normalize based on typical max spread (~30 degrees between fingers)
        max_angle = np.pi / 6  # 30 degrees
        spread = np.mean(angles) / max_angle
        spread = np.clip(spread, 0.0, 1.0)
        
        return spread
    
    def detect_pointing_gesture(self, landmarks: np.ndarray) -> bool:
        """
        Detect if hand is making a pointing gesture.
        
        Args:
            landmarks: Hand landmarks array
            
        Returns:
            True if pointing gesture detected
        """
        # Index finger should be extended, others curled
        finger_curls = [self.calculate_finger_curl(landmarks, i) for i in range(5)]
        
        # Pointing: index extended (low curl), others curled (high curl)
        index_extended = finger_curls[1] < 0.3
        others_curled = all(curl > 0.6 for i, curl in enumerate(finger_curls) if i != 1)
        
        return index_extended and others_curled
    
    def detect_peace_sign(self, landmarks: np.ndarray) -> bool:
        """
        Detect if hand is making a peace/victory sign.
        
        Args:
            landmarks: Hand landmarks array
            
        Returns:
            True if peace sign detected
        """
        finger_curls = [self.calculate_finger_curl(landmarks, i) for i in range(5)]
        
        # Peace sign: index and middle extended, others curled
        index_extended = finger_curls[1] < 0.3
        middle_extended = finger_curls[2] < 0.3
        others_curled = finger_curls[0] > 0.5 and finger_curls[3] > 0.6 and finger_curls[4] > 0.6
        
        return index_extended and middle_extended and others_curled
    
    def detect_thumbs_up(self, landmarks: np.ndarray) -> bool:
        """
        Detect if hand is making a thumbs up gesture.
        
        Args:
            landmarks: Hand landmarks array
            
        Returns:
            True if thumbs up detected
        """
        finger_curls = [self.calculate_finger_curl(landmarks, i) for i in range(5)]
        
        # Thumbs up: thumb extended, others curled
        thumb_extended = finger_curls[0] < 0.4
        others_curled = all(curl > 0.7 for curl in finger_curls[1:])
        
        # Also check thumb is pointing upward
        thumb_tip = landmarks[4]
        thumb_base = landmarks[2]
        thumb_up = thumb_tip[1] < thumb_base[1]  # Y coordinate decreases upward
        
        return thumb_extended and others_curled and thumb_up
    
    def detect_open_palm(self, landmarks: np.ndarray) -> bool:
        """
        Detect if hand is showing open palm.
        
        Args:
            landmarks: Hand landmarks array
            
        Returns:
            True if open palm detected
        """
        openness = self.calculate_hand_openness(landmarks)
        return openness > 0.8
    
    def detect_closed_fist(self, landmarks: np.ndarray) -> bool:
        """
        Detect if hand is making a closed fist.
        
        Args:
            landmarks: Hand landmarks array
            
        Returns:
            True if closed fist detected
        """
        openness = self.calculate_hand_openness(landmarks)
        return openness < 0.2
    
    def calculate_hand_velocity(self, hand_center: np.ndarray, handedness: str) -> float:
        """
        Calculate hand movement velocity.
        
        Args:
            hand_center: Current hand center position
            handedness: 'left' or 'right'
            
        Returns:
            Velocity magnitude
        """
        if self.prev_hand_centers[handedness] is None:
            self.prev_hand_centers[handedness] = hand_center
            return 0.0
        
        # Calculate displacement
        displacement = hand_center - self.prev_hand_centers[handedness]
        velocity = np.linalg.norm(displacement)
        
        # Update previous position
        self.prev_hand_centers[handedness] = hand_center
        
        # Add to movement history
        self.movement_history[handedness].append(velocity)
        
        return velocity
    
    def analyze_gesticulation_intensity(self, handedness: str) -> float:
        """
        Analyze the intensity of hand gesticulation.
        
        Args:
            handedness: 'left' or 'right'
            
        Returns:
            Intensity score (0 = no movement, 1 = high gesticulation)
        """
        if len(self.movement_history[handedness]) < 5:
            return 0.0
        
        # Calculate average movement over recent frames
        avg_movement = np.mean(list(self.movement_history[handedness]))
        
        # Normalize (typical gesticulation has movement of 0.02-0.1 in normalized coords)
        intensity = avg_movement / 0.1
        intensity = np.clip(intensity, 0.0, 1.0)
        
        return intensity
    
    def detect_gesture_phase(self, handedness: str) -> str:
        """
        Detect the phase of gesticulation (preparation, stroke, hold, retraction).
        
        Args:
            handedness: 'left' or 'right'
            
        Returns:
            Gesture phase string
        """
        if len(self.movement_history[handedness]) < 3:
            return "unknown"
        
        recent_velocities = list(self.movement_history[handedness])[-3:]
        avg_velocity = np.mean(recent_velocities)
        
        # Detect acceleration/deceleration
        if len(recent_velocities) >= 3:
            acceleration = recent_velocities[-1] - recent_velocities[-3]
            
            if avg_velocity < 0.01:
                return "hold"
            elif acceleration > 0.005:
                return "preparation"
            elif acceleration < -0.005:
                return "retraction"
            else:
                return "stroke"
        
        return "unknown"
    
    def calculate_hand_symmetry(self, left_metrics: Dict, right_metrics: Dict) -> float:
        """
        Calculate symmetry between left and right hand movements/positions.
        
        Args:
            left_metrics: Left hand metrics
            right_metrics: Right hand metrics
            
        Returns:
            Symmetry score (0 = asymmetric, 1 = perfectly symmetric)
        """
        if not left_metrics or not right_metrics:
            return 0.0
        
        # Compare openness
        openness_diff = abs(left_metrics.get('openness', 0) - right_metrics.get('openness', 0))
        
        # Compare gesture intensity
        intensity_diff = abs(left_metrics.get('gesticulation_intensity', 0) - 
                           right_metrics.get('gesticulation_intensity', 0))
        
        # Calculate symmetry score
        symmetry = 1.0 - (openness_diff * 0.5 + intensity_diff * 0.5)
        
        return symmetry
    
    def analyze_communication_quality(self, hand_metrics: Dict) -> Dict:
        """
        Analyze the quality of non-verbal hand communication.
        
        Args:
            hand_metrics: Dictionary containing hand analysis metrics
            
        Returns:
            Communication quality assessment
        """
        quality_assessment = {
            'overall_score': 0.0,
            'clarity': 0.0,
            'expressiveness': 0.0,
            'appropriateness': 0.0,
            'recommendations': []
        }
        
        # Assess clarity (clear gestures vs ambiguous movements)
        if hand_metrics.get('gesture_confidence', 0) > 0.7:
            quality_assessment['clarity'] = 0.9
        else:
            quality_assessment['clarity'] = 0.5
            quality_assessment['recommendations'].append("Use more defined gestures")
        
        # Assess expressiveness
        intensity = hand_metrics.get('gesticulation_intensity', 0)
        if 0.2 <= intensity <= 0.7:
            quality_assessment['expressiveness'] = 1.0
        elif intensity < 0.2:
            quality_assessment['expressiveness'] = 0.3
            quality_assessment['recommendations'].append("Increase hand movement for emphasis")
        else:
            quality_assessment['expressiveness'] = 0.6
            quality_assessment['recommendations'].append("Reduce excessive hand movements")
        
        # Assess appropriateness (based on gesture types and context)
        if hand_metrics.get('hands_in_frame', False):
            quality_assessment['appropriateness'] = 0.8
        else:
            quality_assessment['appropriateness'] = 0.5
            quality_assessment['recommendations'].append("Keep hands visible during communication")
        
        # Calculate overall score
        quality_assessment['overall_score'] = np.mean([
            quality_assessment['clarity'],
            quality_assessment['expressiveness'],
            quality_assessment['appropriateness']
        ])
        
        return quality_assessment
    
    def process_frame(self, frame_rgb: np.ndarray) -> Optional[Dict]:
        """
        Process a single frame for hand tracking and gesture analysis.
        
        Args:
            frame_rgb: Input frame in RGB format
            
        Returns:
            Dictionary containing comprehensive hand metrics or None if no hands detected
        """
        # Run MediaPipe hand detection
        results = self.hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            # Update history to reflect no hands
            self.prev_hand_centers = {'left': None, 'right': None}
            return {
                'hands_detected': False,
                'hands_in_frame': False,
                'num_hands': 0,
                'hand_metrics': {}
            }
        
        # Process detected hands
        h, w = frame_rgb.shape[:2]
        hand_metrics = {}
        hands_info = {
            'hands_detected': True,
            'hands_in_frame': True,
            'num_hands': len(results.multi_hand_landmarks),
            'hand_metrics': {},
            'raw_landmarks': {},
            'hand_landmarks': results.multi_hand_landmarks,
            'handedness': results.multi_handedness
        }
        
        for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, 
                                                               results.multi_handedness)):
            # Determine hand side
            hand_label = handedness.classification[0].label.lower()
            
            # Convert landmarks to numpy array
            landmarks_array = np.array([(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark])
            hands_info['raw_landmarks'][hand_label] = landmarks_array
            
            # Calculate hand center
            hand_center = np.mean(landmarks_array, axis=0)
            
            # Calculate all hand metrics
            openness = self.calculate_hand_openness(landmarks_array)
            finger_spread = self.calculate_finger_spread(landmarks_array)
            velocity = self.calculate_hand_velocity(hand_center[:2], hand_label)
            
            # Detect specific gestures
            gestures_detected = {
                'pointing': self.detect_pointing_gesture(landmarks_array),
                'peace_sign': self.detect_peace_sign(landmarks_array),
                'thumbs_up': self.detect_thumbs_up(landmarks_array),
                'open_palm': self.detect_open_palm(landmarks_array),
                'closed_fist': self.detect_closed_fist(landmarks_array)
            }
            
            # Get primary gesture
            active_gestures = [g for g, detected in gestures_detected.items() if detected]
            primary_gesture = active_gestures[0] if active_gestures else 'none'
            
            # Calculate gesture confidence
            gesture_confidence = 1.0 if primary_gesture != 'none' else 0.0
            
            # Analyze gesticulation
            gesticulation_intensity = self.analyze_gesticulation_intensity(hand_label)
            gesture_phase = self.detect_gesture_phase(hand_label)
            
            # Calculate individual finger states
            finger_states = {}
            finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
            for i, name in enumerate(finger_names):
                curl = self.calculate_finger_curl(landmarks_array, i)
                finger_states[name] = {
                    'curl': curl,
                    'extended': curl < 0.3
                }
            
            # Store metrics for this hand
            hand_metrics[hand_label] = {
                'openness': openness,
                'finger_spread': finger_spread,
                'velocity': velocity,
                'gesticulation_intensity': gesticulation_intensity,
                'gesture_phase': gesture_phase,
                'primary_gesture': primary_gesture,
                'gesture_confidence': gesture_confidence,
                'gestures_detected': gestures_detected,
                'finger_states': finger_states,
                'hand_center': hand_center.tolist(),
                'is_dominant': idx == 0  # First detected hand is usually dominant
            }
            
            # Add to history
            self.hand_history[hand_label].append(hand_metrics[hand_label])
        
        # Calculate symmetry if both hands detected
        if 'left' in hand_metrics and 'right' in hand_metrics:
            symmetry = self.calculate_hand_symmetry(
                hand_metrics['left'], 
                hand_metrics['right']
            )
            hands_info['hand_symmetry'] = symmetry
        else:
            hands_info['hand_symmetry'] = 0.0
        
        # Analyze overall communication quality
        if hand_metrics:
            # Use dominant hand or first available hand
            dominant_hand = next(iter(hand_metrics.values()))
            hands_info['communication_quality'] = self.analyze_communication_quality(dominant_hand)
        
        hands_info['hand_metrics'] = hand_metrics
        
        return hands_info
    
    def draw_hand_visualization(self, 
                              frame: np.ndarray, 
                              hand_info: Dict,
                              debug: bool = False) -> np.ndarray:
        """
        Draw hand tracking visualization on frame.
        
        Args:
            frame: Input frame
            hand_info: Hand tracking information
            debug: Whether to show debug visualization
            
        Returns:
            Frame with visualization
        """
        vis_frame = frame.copy()
        
        if not hand_info['hands_detected']:
            return vis_frame
        
        # Draw hand landmarks and connections
        if debug and 'hand_landmarks' in hand_info:
            for hand_landmarks, handedness in zip(hand_info['hand_landmarks'], 
                                                 hand_info['handedness']):
                # Draw hand skeleton
                self.mp_drawing.draw_landmarks(
                    vis_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Get hand label
                hand_label = handedness.classification[0].label
                
                # Draw hand label
                if hand_label.lower() in hand_info['hand_metrics']:
                    metrics = hand_info['hand_metrics'][hand_label.lower()]
                    
                    # Find wrist position
                    wrist_landmark = hand_landmarks.landmark[0]
                    wrist_x = int(wrist_landmark.x * vis_frame.shape[1])
                    wrist_y = int(wrist_landmark.y * vis_frame.shape[0])
                    
                    # Draw gesture info
                    gesture_text = f"{hand_label}: {metrics['primary_gesture']}"
                    cv2.putText(vis_frame, gesture_text,
                               (wrist_x - 50, wrist_y + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    # Draw openness indicator
                    openness = metrics['openness']
                    openness_color = (0, int(255 * openness), int(255 * (1 - openness)))
                    cv2.putText(vis_frame, f"Open: {openness:.2f}",
                               (wrist_x - 50, wrist_y + 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, openness_color, 2)
        
        return vis_frame