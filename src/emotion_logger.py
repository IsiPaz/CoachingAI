import cv2
import json
import numpy as np
import time
from typing import Dict, Optional


class EmotionLogger:
    """
    Handles logging, debugging, and visualization for emotion recognition.
    """
    
    def __init__(self, 
                 debug: bool = False,
                 show_fps: bool = False,
                 show_circumplex: bool = False):
        """
        Initialize the emotion logger.
        
        Args:
            debug: Whether to enable debug mode
            show_fps: Whether to display FPS counter
            show_circumplex: Whether to display circumplex visualization
        """
        self.debug = debug
        self.show_fps = show_fps
        self.show_circumplex = show_circumplex
        
        # Debug counters
        self.debug_frame_counter = 0
        self.debug_print_interval = 30
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0

        # Last state to allow saving debug JSON
        self.last_face_bbox = None
        self.last_emotion_info = None
        self.last_iris_info = None
        self.last_pose_info = None
        self.last_distance_info = None
        self.last_hand_info = None
        self.last_device = "unknown"

    def set_last_debug_state(self,
                            face_bbox: Optional[np.ndarray],
                            emotion_info: Optional[Dict],
                            iris_info: Optional[Dict],
                            pose_info: Optional[Dict],
                            distance_info: Optional[Dict],
                            hand_info: Optional[Dict],
                            device: str) -> None:
        """Store the latest debug state for screenshot JSON logging."""
        self.last_face_bbox = face_bbox
        self.last_emotion_info = emotion_info
        self.last_iris_info = iris_info
        self.last_pose_info = pose_info
        self.last_distance_info = distance_info
        self.last_hand_info = hand_info
        self.last_device = device

    def update_fps(self) -> None:
        """Update FPS counter."""
        if self.show_fps:
            self.fps_counter += 1
            if self.fps_counter >= 30:
                current_time = time.time()
                self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
                self.fps_counter = 0
                self.fps_start_time = current_time
                
    def increment_frame_counter(self) -> None:
        """Increment debug frame counter."""
        self.debug_frame_counter += 1
        
    def toggle_debug_interval(self) -> None:
        """Toggle debug print interval between 1 and 30 frames."""
        self.debug_print_interval = 1 if self.debug_print_interval == 30 else 30
        print(f"Debug print interval changed to: {self.debug_print_interval} frames")
        

    def print_debug_info(self, 
                        face_bbox: Optional[np.ndarray], 
                        emotion_info: Optional[Dict],
                        iris_info: Optional[Dict],
                        pose_info: Optional[Dict],
                        distance_info: Optional[Dict],
                        hand_info: Optional[Dict],
                        device: str) -> None:
        """
        Print detailed debug information to console.
        
        Args:
            face_bbox: Face bounding box coordinates
            emotion_info: Processed emotion information
            iris_info: Raw iris tracking information
            pose_info: Pose tracking information
            distance_info: Distance measurement information
            hand_info: Hand tracking information
            device: Device being used for inference
        """
        if not self.debug:
            return
            
        # Check if we should print this frame
        if self.debug_frame_counter % self.debug_print_interval != 0:
            return
            
        print(f"\n{'='*60}")
        print(f"DEBUG INFO - Frame {self.debug_frame_counter}")
        print(f"{'='*60}")
        
        # Face detection info
        if face_bbox is not None:
            print(f"Face detected at: ({face_bbox[0]}, {face_bbox[1]}) - ({face_bbox[2]}, {face_bbox[3]})")
            face_width = face_bbox[2] - face_bbox[0]
            face_height = face_bbox[3] - face_bbox[1]
            print(f"Face size: {face_width}x{face_height} pixels")
        else:
            print("No face detected")
            return
            
        if emotion_info is None:
            print("No emotion analysis available")
            return
            
        # Emotion probabilities
        predicted_emotion = emotion_info["predicted_emotion"]
        confidence = emotion_info["confidence"]
        
        print(f"\nEMOTION ANALYSIS:")
        print(f"Predicted emotion: {predicted_emotion} (confidence: {confidence:.3f})")
        
        print(f"\nAll emotion probabilities:")
        for emotion, prob in emotion_info["all_probabilities"].items():
            is_predicted = emotion == predicted_emotion
            print(f"  {emotion:>10}: {prob:.3f} {'<-- PREDICTED' if is_predicted else ''}")
            
        # Valence and Arousal
        valence = emotion_info["valence"]
        arousal = emotion_info["arousal"]
        
        print(f"\nVALENCE-AROUSAL ANALYSIS:")
        print(f"Valence: {valence:+.3f} ({'Positive' if valence > 0 else 'Negative' if valence < 0 else 'Neutral'})")
        print(f"Arousal:  {arousal:+.3f} ({'High' if arousal > 0 else 'Low' if arousal < 0 else 'Neutral'})")
        
        # Distance Tracking Raw Values
        if distance_info is not None:
            print(f"\n{'='*60}")
            print(f"CAMERA DISTANCE TRACKING:")
            print(f"{'='*60}")
            
            print(f"\nDISTANCE MEASUREMENTS:")
            print(f"  Current Distance:       {distance_info['distance_cm']:.1f} cm")
            print(f"  Raw Distance:           {distance_info['raw_distance_cm']:.1f} cm")
            print(f"  Measurement Method:     {distance_info['measurement_method'].upper()}")
            print(f"  Face Size in Frame:     {distance_info['face_size_percentage']:.1f}%")
            print(f"  Calibrated:             {'Yes' if distance_info['is_calibrated'] else 'No'}")
            
            print(f"\nDISTANCE STATUS:")
            print(f"  Status:                 {distance_info['distance_status']}")
            print(f"  Quality:                {distance_info['distance_quality'].upper()}")
            print(f"  Recommendation:         {distance_info['recommendation']}")
            print(f"  Optimal Range:          {distance_info['optimal_range']}")
        
        # Iris and Eye Tracking Raw Values
        if iris_info is not None:
            print(f"\n{'='*60}")
            print(f"IRIS AND EYE TRACKING RAW VALUES:")
            print(f"{'='*60}")
            
            # Eye Aperture Values
            print(f"\nEYE APERTURE (0=closed, ~0.3=normal):")
            print(f"  Left Eye Aperture:  {iris_info['left_eye_aperture']:.4f}")
            print(f"  Right Eye Aperture: {iris_info['right_eye_aperture']:.4f}")
            print(f"  Average Aperture:   {iris_info['average_eye_aperture']:.4f}")
            
            # Iris Position Values
            iris_pos = iris_info['iris_position']
            print(f"\n 👀IRIS POSITION (normalized -1 to +1):")
            print(f"  Left Iris:")
            print(f"    Horizontal Offset: {iris_pos['left_iris_horizontal_offset']:+.4f} ({'Right' if iris_pos['left_iris_horizontal_offset'] > 0 else 'Left' if iris_pos['left_iris_horizontal_offset'] < 0 else 'Center'})")
            print(f"    Vertical Offset:   {iris_pos['left_iris_vertical_offset']:+.4f} ({'Down' if iris_pos['left_iris_vertical_offset'] > 0 else 'Up' if iris_pos['left_iris_vertical_offset'] < 0 else 'Center'})")
            print(f"    Centering Score:   {iris_pos['left_iris_centering']:.4f} (0=centered, 1=edge)")
            
            print(f"  Right Iris:")
            print(f"    Horizontal Offset: {iris_pos['right_iris_horizontal_offset']:+.4f} ({'Right' if iris_pos['right_iris_horizontal_offset'] > 0 else 'Left' if iris_pos['right_iris_horizontal_offset'] < 0 else 'Center'})")
            print(f"    Vertical Offset:   {iris_pos['right_iris_vertical_offset']:+.4f} ({'Down' if iris_pos['right_iris_vertical_offset'] > 0 else 'Up' if iris_pos['right_iris_vertical_offset'] < 0 else 'Center'})")
            print(f"    Centering Score:   {iris_pos['right_iris_centering']:.4f} (0=centered, 1=edge)")
            
            print(f"  Average Values:")
            print(f"    Avg Horizontal:    {iris_pos['average_horizontal_offset']:+.4f}")
            print(f"    Avg Vertical:      {iris_pos['average_vertical_offset']:+.4f}")
            print(f"    Avg Centering:     {iris_pos['average_centering']:.4f}")
            
            # Blink Information
            print(f"\nBLINK TRACKING:")
            print(f"  Currently Blinking:     {'Yes' if iris_info['is_blinking'] else 'No'}")
            print(f"  Total Blinks:           {iris_info['total_blinks']}")
            print(f"  Frames Since Last Blink: {iris_info['frames_since_last_blink']}")
            
            # Eye Metrics
            metrics = iris_info['eye_metrics']
            print(f"\nEYE METRICS (last 5 seconds):")
            print(f"  Blink Count:            {metrics['blink_count_last_5_sec']}")
            print(f"  Average Eye Aperture:   {metrics['average_ear_last_5_sec']:.4f}")
            print(f"  Eye Closure Percentage: {metrics['eye_closure_percentage']:.1f}%")
            print(f"  Longest Closure:        {metrics['longest_closure_frames']} frames")

        # Pose and Body Tracking Raw Values
        if pose_info is not None:
            print(f"\n{'='*60}")
            print(f"POSE AND BODY TRACKING RAW VALUES:")
            print(f"{'='*60}")
            
            # Body posture metrics
            posture = pose_info['posture_metrics']

            score = posture['overall_posture_score']
            if score > 0.68:
                quality_text = "Excellent"
            elif score > 0.48:
                quality_text = "Good"
            else:
                quality_text = "Needs Improvement"
        
            pose_info["posture_metrics"]["posture_quality_text"] = quality_text

            print(f"\nBODY POSTURE ANALYSIS:")
            print(f"  Trunk Inclination:")
            print(f"    Forward/Backward: {posture['trunk_inclination_fb']:+.2f}° ({'Forward' if posture['trunk_inclination_fb'] > 0 else 'Backward' if posture['trunk_inclination_fb'] < 0 else 'Neutral'})")
            print(f"    Left/Right Lean:  {posture['trunk_inclination_lr']:+.2f}° ({'Right' if posture['trunk_inclination_lr'] > 0 else 'Left' if posture['trunk_inclination_lr'] < 0 else 'Neutral'})")
            
            print(f"\n  Shoulder Symmetry:")
            print(f"    Height Difference: {posture['shoulder_asymmetry']:+.2f}° ({'Right Higher' if posture['shoulder_asymmetry'] > 0 else 'Left Higher' if posture['shoulder_asymmetry'] < 0 else 'Level'})")
            print(f"    Symmetry Score:    {posture['shoulder_symmetry_score']:.3f} (1.0=perfect)")
            print(f"    Symmetry Status:    {posture['shoulder_symmetry_status']} {'✓' if posture['shoulder_symmetry_status'] == 'Correct' else '✗'}")

            print(f"\n  Head Orientation:")
            print(f"    Head Tilt:         {posture['head_tilt']:+.2f}° ({'Right' if posture['head_tilt'] > 0 else 'Left' if posture['head_tilt'] < 0 else 'Neutral'})")
            print(f"    Head Turn:         {posture['head_turn']:+.2f}° ({'Right' if posture['head_turn'] > 0 else 'Left' if posture['head_turn'] < 0 else 'Forward'})")
            
            print(f"\n  Estimated Orientation:")
            print(f"    Person is facing: {posture.get('orientation', 'Unknown')}")

            print(f"\n  OVERALL POSTURE:")
            print(f"    Overall Alignment: {posture['overall_posture_score']:.3f}")

        # ========== HAND TRACKING AND GESTURE ANALYSIS ==========
        if hand_info is not None:
            print(f"\n{'='*60}")
            print(f"HAND TRACKING AND GESTURE ANALYSIS:")
            print(f"{'='*60}")
            
            print(f"\nHAND DETECTION STATUS:")
            print(f"  Hands Detected:         {'YES' if hand_info.get('hands_detected', False) else 'NO'}")
            print(f"  Hands in Frame:         {'YES' if hand_info.get('hands_in_frame', False) else 'NO'}")
            print(f"  Number of Hands:        {hand_info.get('num_hands', 0)}")
            
            # Individual hand analysis
            hand_metrics = hand_info.get('hand_metrics', {})
            if hand_metrics:
                for hand_side, metrics in hand_metrics.items():
                    print(f"\n  {hand_side.upper()} HAND ANALYSIS:")
                    print(f"    Hand State:           {metrics.get('hand_state', 'unknown').upper()}")
                    print(f"    Openness:             {metrics.get('openness', 0):.3f} (0=closed, 1=open)")
                    print(f"    Finger Spread:        {metrics.get('finger_spread', 0):.3f} (0=together, 1=spread)")
                    print(f"    Primary Gesture:      {metrics.get('primary_gesture', 'none').upper()}")
                    print(f"    Gesture Confidence:   {metrics.get('gesture_confidence', 0):.3f}")
                    print(f"    Movement Velocity:    {metrics.get('velocity', 0):.4f}")
                    print(f"    Gesticulation Intensity: {metrics.get('gesticulation_intensity', 0):.3f}")
                    print(f"    Gesture Phase:        {metrics.get('gesture_phase', 'unknown').upper()}")
                    print(f"    Is Dominant Hand:     {'YES' if metrics.get('is_dominant', False) else 'NO'}")
                    
                    # Specific gestures detected
                    gestures = metrics.get('gestures_detected', {})
                    active_gestures = [g for g, detected in gestures.items() if detected]
                    if active_gestures:
                        print(f"    Active Gestures:      {', '.join(active_gestures).upper()}")
                    else:
                        print(f"    Active Gestures:      NONE")
                    
                    # Finger states
                    finger_states = metrics.get('finger_states', {})
                    if finger_states:
                        print(f"    Finger States:")
                        for finger_name, state in finger_states.items():
                            curl = state.get('curl', 0)
                            extended = state.get('extended', False)
                            print(f"      {finger_name.capitalize():>6}: {'Extended' if extended else 'Curled'} (curl: {curl:.3f})")
            
            # Hand symmetry
            if hand_info.get('hand_symmetry') is not None:
                symmetry = hand_info['hand_symmetry']
                print(f"\n  HAND SYMMETRY:")
                print(f"    Symmetry Score:       {symmetry:.3f} (0=asymmetric, 1=symmetric)")
                if symmetry > 0.7:
                    symmetry_status = "SYMMETRIC"
                elif symmetry > 0.4:
                    symmetry_status = "MODERATELY SYMMETRIC"
                else:
                    symmetry_status = "ASYMMETRIC"
                print(f"    Symmetry Status:      {symmetry_status}")
            
            # Face interference detailed analysis
            interference = hand_info.get('face_interference', {})
            if interference:
                print(f"\n  FACE INTERFERENCE ANALYSIS:")
                print(f"    Currently Interfering:  {'YES' if interference.get('is_interfering', False) else 'NO'}")
                print(f"    Interference Score:     {interference.get('interference_score', 0):.3f} (0=no overlap, 1=complete overlap)")
                print(f"    Duration:               {interference.get('duration', 0):.1f} seconds")
                print(f"    Sustained Interference: {'YES - FACE COVERED!' if interference.get('sustained_interference', False) else 'NO'}")
                
                interfering_hands = interference.get('interfering_hands', [])
                if interfering_hands:
                    hand_names = []
                    for idx in interfering_hands:
                        if idx == 0 and 'left' in hand_metrics:
                            hand_names.append('LEFT')
                        elif idx == 1 and 'right' in hand_metrics:
                            hand_names.append('RIGHT')
                        else:
                            hand_names.append(f'HAND_{idx}')
                    print(f"    Interfering Hands:      {', '.join(hand_names)}")
                else:
                    print(f"    Interfering Hands:      NONE")
        else:
            print(f"\n{'='*60}")
            print(f"HAND TRACKING: NO HAND DATA AVAILABLE")
            print(f"{'='*60}")

        # Emotional quadrant
        quadrant = self._get_emotional_quadrant(valence, arousal)
        print(f"\nEmotional quadrant: {quadrant}")
        
        # Performance info
        print(f"\nPERFORMANCE:")
        print(f"Current FPS: {self.current_fps:.1f}")
        print(f"Device: {device}")
        
        print(f"{'='*60}\n")

        
    def _get_emotional_quadrant(self, valence: float, arousal: float) -> str:
        """
        Get emotional quadrant based on valence and arousal.
        
        Args:
            valence: Valence value [-1, 1]
            arousal: Arousal value [-1, 1]
            
        Returns:
            String description of emotional quadrant
        """
        if valence > 0 and arousal > 0:
            return "High Arousal + Positive Valence (Excited/Happy)"
        elif valence > 0 and arousal < 0:
            return "Low Arousal + Positive Valence (Calm/Content)"
        elif valence < 0 and arousal > 0:
            return "High Arousal + Negative Valence (Stressed/Angry)"
        elif valence < 0 and arousal < 0:
            return "Low Arousal + Negative Valence (Sad/Depressed)"
        else:
            return "Neutral state"
            
    def create_circumplex_visualization(self, valence: float, arousal: float) -> np.ndarray:
        """
        Create valence-arousal circumplex visualization.
        
        Args:
            valence: Valence value [-1, 1]
            arousal: Arousal value [-1, 1]
            
        Returns:
            Circumplex visualization image
        """
        circumplex_size = 300
        
        # Create a simple circumplex background
        circumplex = np.zeros((circumplex_size, circumplex_size, 3), dtype=np.uint8)
        circumplex[:] = (50, 50, 50)  # Dark gray background
        
        # Draw axes
        cv2.line(circumplex, (0, circumplex_size//2), (circumplex_size, circumplex_size//2), (100, 100, 100), 2)
        cv2.line(circumplex, (circumplex_size//2, 0), (circumplex_size//2, circumplex_size), (100, 100, 100), 2)
        
        # Draw circle
        cv2.circle(circumplex, (circumplex_size//2, circumplex_size//2), circumplex_size//2-10, (100, 100, 100), 2)
        
        # Plot valence-arousal point
        x = int((valence + 1.0) / 2.0 * circumplex_size)
        y = int((1.0 - arousal) / 2.0 * circumplex_size)  # Flip Y axis
        
        cv2.circle(circumplex, (x, y), 8, (0, 0, 255), -1)
        
        # Add labels
        cv2.putText(circumplex, "Arousal", (circumplex_size//2-30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(circumplex, "Valence", (circumplex_size-60, circumplex_size//2+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return circumplex
        
    def create_visualization(self, 
                            frame_bgr: np.ndarray,
                            face_bbox: Optional[np.ndarray] = None,
                            emotion_info: Optional[Dict] = None,
                            iris_info: Optional[Dict] = None,
                            distance_info: Optional[Dict] = None,
                            pose_info: Optional[Dict] = None,
                            hand_info: Optional[Dict] = None) -> np.ndarray:

        """
        Create the final visualization with emotion, iris, distance, and hand information.
        
        Args:
            frame_bgr: Input frame in BGR format
            face_bbox: Face bounding box coordinates
            emotion_info: Processed emotion information
            iris_info: Raw iris tracking information
            distance_info: Distance measurement information
            pose_info: Pose tracking information
            hand_info: Hand tracking information
            
        Returns:
            Visualization frame
        """
        vis_frame = frame_bgr.copy()
        
        if face_bbox is not None and emotion_info is not None:
            # Draw face bounding box
            cv2.rectangle(vis_frame, 
                         (face_bbox[0], face_bbox[1]), 
                         (face_bbox[2], face_bbox[3]), 
                         (0, 255, 0), 3)
            
            # Draw emotion text
            emotion_text = f"{emotion_info['predicted_emotion']}: {emotion_info['confidence']:.2f}"
            cv2.putText(vis_frame, emotion_text,
                       (face_bbox[0], face_bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw valence/arousal info
            va_text = f"V: {emotion_info['valence']:.2f}, A: {emotion_info['arousal']:.2f}"
            cv2.putText(vis_frame, va_text,
                       (face_bbox[0], face_bbox[3] + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Create circumplex visualization (only if option enabled)
            if self.show_circumplex:
                circumplex = self.create_circumplex_visualization(
                    emotion_info['valence'], emotion_info['arousal']
                )
                
                # Overlay circumplex on frame (top-right corner)
                overlay_size = min(200, vis_frame.shape[1]//4)
                circumplex_resized = cv2.resize(circumplex, (overlay_size, overlay_size))
                
                y_offset = 10
                x_offset = vis_frame.shape[1] - overlay_size - 10
                
                # Add semi-transparent overlay
                overlay_region = vis_frame[y_offset:y_offset+overlay_size, x_offset:x_offset+overlay_size]
                blended = cv2.addWeighted(overlay_region, 0.3, circumplex_resized, 0.7, 0)
                vis_frame[y_offset:y_offset+overlay_size, x_offset:x_offset+overlay_size] = blended
        
        # Draw distance information if available
        if distance_info is not None:
            distance_text = f"Distance: {distance_info['distance_cm']:.0f}cm - {distance_info['distance_status']}"
            font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            (text_w, text_h), _ = cv2.getTextSize(distance_text, font, scale, thickness)
            x, y = 10, 25

            # Draw white background rectangle
            cv2.rectangle(vis_frame, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), (255, 255, 255), -1)
            # Draw colored text on top
            cv2.putText(vis_frame, distance_text, (x, y), font, scale, distance_info['status_color'], thickness)

                    
        # Draw iris information if available
        if iris_info is not None:           
            # Draw blink count
            blink_text = f"Blinks: {iris_info['total_blinks']}"
            f, s, t = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            (w, h), _ = cv2.getTextSize(blink_text, f, s, t)
            org = (10, 70 + h)
            cv2.rectangle(vis_frame, (org[0]-5, org[1]-h-5), (org[0]+w+5, org[1]+5), (255,255,255), -1)
            cv2.putText(vis_frame, blink_text, org, f, s, (0,0,0), t)
                    
        # Draw hand information if available
        x, y = 10, vis_frame.shape[0] - 307 + 18

        if hand_info and hand_info.get('face_interference', {}).get('sustained_interference', False):
            interference = hand_info['face_interference']
            alert_text = "FACE COVERED"
            duration_text = f"Duration: {interference['duration']:.1f}s"
            font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2

            (alert_w, alert_h), alert_bl = cv2.getTextSize(alert_text, font, scale, thickness)
            (dur_w, dur_h), dur_bl = cv2.getTextSize(duration_text, font, scale, thickness)

            bg_w = max(alert_w, dur_w) + 10
            bg_h = alert_h + alert_bl + dur_h + dur_bl + 15

            cv2.rectangle(vis_frame, 
                        (x - 5, y - bg_h + dur_bl), 
                        (x - 5 + bg_w, y + dur_bl + 5), 
                        (255, 255, 255), cv2.FILLED)

            cv2.putText(vis_frame, alert_text, (x, y - dur_h - 5), font, scale, (0, 0, 255), thickness)
            cv2.putText(vis_frame, duration_text, (x, y), font, scale, (0, 0, 255), thickness)


        if self.debug and pose_info is not None:
            orientation = pose_info['posture_metrics'].get("orientation", "Unknown")
            f, s, t = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            blink_h = cv2.getTextSize(f"Blinks: {iris_info['total_blinks']}", f, s, t)[0][1]
            org = (10, 70 + blink_h + 15 + blink_h) 

            # Force color: green if frontal, else red (any other case)
            color = (0, 255, 0) if orientation.lower() == "frontal" else (0, 0, 255)

            (w, h), _ = cv2.getTextSize(f"Orientation: {orientation}", f, s, t)
            cv2.rectangle(vis_frame, (org[0]-5, org[1]-h-5), (org[0]+w+5, org[1]+5), (255,255,255), -1)
            cv2.putText(vis_frame, f"Orientation: {orientation}", org, f, s, color, t)

            # Shoulders
            shoulder_status = pose_info['posture_metrics'].get('shoulder_symmetry_status', 'Unknown')
            shoulder_text = f"Shoulder Sym: {shoulder_status}"
            
            # Use the same font variables already defined
            font = f  # cv2.FONT_HERSHEY_SIMPLEX
            scale = s  # 0.6
            thickness = t  # 2
            
            # Get text size
            (text_w, text_h), _ = cv2.getTextSize(shoulder_text, font, scale, thickness)
            
            # Position - below orientation text
            x_pos = 10
            y_pos = org[1] + h + 16  # Position below orientation with some spacing
            
            # Draw white background
            cv2.rectangle(vis_frame, 
                        (x_pos - 5, y_pos - text_h - 5), 
                        (x_pos + text_w + 5, y_pos + 5), 
                        (255, 255, 255), -1)
            
            # Set color based on status
            text_color = (0, 255, 0) if shoulder_status == "Correct" else (0, 0, 255)
            
            # Draw text
            cv2.putText(vis_frame, shoulder_text, (x_pos, y_pos), 
                    font, scale, text_color, thickness)


        # Position below orientation if debug, else use fixed position
        if self.debug and iris_info is not None:
            blink_h = cv2.getTextSize(f"Blinks: {iris_info['total_blinks']}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][1]
            y_pos = 100 + 15 + blink_h + 17 + blink_h + 35  # Below orientation
        else:
            y_pos = 150  # Fixed position
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2

        # Add debug indicator
        if self.debug:
            text = "DEBUG MODE"; f, s, t = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            (w, h), _ = cv2.getTextSize(text, f, s, t)
            org = (vis_frame.shape[1] - w - 10, 10 + h)
            cv2.rectangle(vis_frame, (org[0]-5, org[1]-h-5), (org[0]+w+5, org[1]+5), (255,255,255), -1)
            cv2.putText(vis_frame, text, org, f, s, (255,0,0), t)

        
        return vis_frame
        

    def save_debug_txt(self, txt_filename: str) -> None:
        """Save the latest debug information as a formatted text file."""
        
        try:
            with open(txt_filename, "w", encoding='utf-8') as f:
                f.write(f"{'='*60}\n")
                f.write(f"DEBUG INFO - SCREENSHOT CAPTURE\n")
                f.write(f"Timestamp: {int(time.time())}\n")
                f.write(f"{'='*60}\n")
                
                # Face detection info
                if self.last_face_bbox is not None:
                    f.write(f"Face detected at: ({self.last_face_bbox[0]}, {self.last_face_bbox[1]}) - ({self.last_face_bbox[2]}, {self.last_face_bbox[3]})\n")
                    face_width = self.last_face_bbox[2] - self.last_face_bbox[0]
                    face_height = self.last_face_bbox[3] - self.last_face_bbox[1]
                    f.write(f"Face size: {face_width}x{face_height} pixels\n")
                else:
                    f.write("No face detected\n")
                    return
                    
                if self.last_emotion_info is None:
                    f.write("No emotion analysis available\n")
                    return
                    
                # Emotion probabilities
                predicted_emotion = self.last_emotion_info["predicted_emotion"]
                confidence = self.last_emotion_info["confidence"]
                
                f.write(f"\nEMOTION ANALYSIS:\n")
                f.write(f"Predicted emotion: {predicted_emotion} (confidence: {confidence:.3f})\n")
                
                f.write(f"\nAll emotion probabilities:\n")
                for emotion, prob in self.last_emotion_info["all_probabilities"].items():
                    is_predicted = emotion == predicted_emotion
                    f.write(f"  {emotion:>10}: {prob:.3f} {'<-- PREDICTED' if is_predicted else ''}\n")
                    
                # Valence and Arousal
                valence = self.last_emotion_info["valence"]
                arousal = self.last_emotion_info["arousal"]
                
                f.write(f"\nVALENCE-AROUSAL ANALYSIS:\n")
                f.write(f"Valence: {valence:+.3f} ({'Positive' if valence > 0 else 'Negative' if valence < 0 else 'Neutral'})\n")
                f.write(f"Arousal:  {arousal:+.3f} ({'High' if arousal > 0 else 'Low' if arousal < 0 else 'Neutral'})\n")
                
                # Distance Tracking Raw Values
                if self.last_distance_info is not None:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"CAMERA DISTANCE TRACKING:\n")
                    f.write(f"{'='*60}\n")
                    
                    f.write(f"\nDISTANCE MEASUREMENTS:\n")
                    f.write(f"  Current Distance:       {self.last_distance_info['distance_cm']:.1f} cm\n")
                    f.write(f"  Raw Distance:           {self.last_distance_info['raw_distance_cm']:.1f} cm\n")
                    f.write(f"  Measurement Method:     {self.last_distance_info['measurement_method'].upper()}\n")
                    f.write(f"  Face Size in Frame:     {self.last_distance_info['face_size_percentage']:.1f}%\n")
                    f.write(f"  Calibrated:             {'Yes' if self.last_distance_info['is_calibrated'] else 'No'}\n")
                    
                    f.write(f"\nDISTANCE STATUS:\n")
                    f.write(f"  Status:                 {self.last_distance_info['distance_status']}\n")
                    f.write(f"  Quality:                {self.last_distance_info['distance_quality'].upper()}\n")
                    f.write(f"  Recommendation:         {self.last_distance_info['recommendation']}\n")
                    f.write(f"  Optimal Range:          {self.last_distance_info['optimal_range']}\n")
                
                # Iris and Eye Tracking Raw Values
                if self.last_iris_info is not None:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"IRIS AND EYE TRACKING RAW VALUES:\n")
                    f.write(f"{'='*60}\n")
                    
                    # Eye Aperture Values
                    f.write(f"\nEYE APERTURE (0=closed, ~0.3=normal):\n")
                    f.write(f"  Left Eye Aperture:  {self.last_iris_info['left_eye_aperture']:.4f}\n")
                    f.write(f"  Right Eye Aperture: {self.last_iris_info['right_eye_aperture']:.4f}\n")
                    f.write(f"  Average Aperture:   {self.last_iris_info['average_eye_aperture']:.4f}\n")
                    
                    # Iris Position Values
                    iris_pos = self.last_iris_info['iris_position']
                    f.write(f"\n👀IRIS POSITION (normalized -1 to +1):\n")
                    f.write(f"  Left Iris:\n")
                    f.write(f"    Horizontal Offset: {iris_pos['left_iris_horizontal_offset']:+.4f} ({'Right' if iris_pos['left_iris_horizontal_offset'] > 0 else 'Left' if iris_pos['left_iris_horizontal_offset'] < 0 else 'Center'})\n")
                    f.write(f"    Vertical Offset:   {iris_pos['left_iris_vertical_offset']:+.4f} ({'Down' if iris_pos['left_iris_vertical_offset'] > 0 else 'Up' if iris_pos['left_iris_vertical_offset'] < 0 else 'Center'})\n")
                    f.write(f"    Centering Score:   {iris_pos['left_iris_centering']:.4f} (0=centered, 1=edge)\n")
                    
                    f.write(f"  Right Iris:\n")
                    f.write(f"    Horizontal Offset: {iris_pos['right_iris_horizontal_offset']:+.4f} ({'Right' if iris_pos['right_iris_horizontal_offset'] > 0 else 'Left' if iris_pos['right_iris_horizontal_offset'] < 0 else 'Center'})\n")
                    f.write(f"    Vertical Offset:   {iris_pos['right_iris_vertical_offset']:+.4f} ({'Down' if iris_pos['right_iris_vertical_offset'] > 0 else 'Up' if iris_pos['right_iris_vertical_offset'] < 0 else 'Center'})\n")
                    f.write(f"    Centering Score:   {iris_pos['right_iris_centering']:.4f} (0=centered, 1=edge)\n")
                    
                    f.write(f"  Average Values:\n")
                    f.write(f"    Avg Horizontal:    {iris_pos['average_horizontal_offset']:+.4f}\n")
                    f.write(f"    Avg Vertical:      {iris_pos['average_vertical_offset']:+.4f}\n")
                    f.write(f"    Avg Centering:     {iris_pos['average_centering']:.4f}\n")
                    
                    # Blink Information
                    f.write(f"\nBLINK TRACKING:\n")
                    f.write(f"  Currently Blinking:     {'Yes' if self.last_iris_info['is_blinking'] else 'No'}\n")
                    f.write(f"  Total Blinks:           {self.last_iris_info['total_blinks']}\n")
                    f.write(f"  Frames Since Last Blink: {self.last_iris_info['frames_since_last_blink']}\n")
                    
                    # Eye Metrics
                    metrics = self.last_iris_info['eye_metrics']
                    f.write(f"\nEYE METRICS (last 5 seconds):\n")
                    f.write(f"  Blink Count:            {metrics['blink_count_last_5_sec']}\n")
                    f.write(f"  Average Eye Aperture:   {metrics['average_ear_last_5_sec']:.4f}\n")
                    f.write(f"  Eye Closure Percentage: {metrics['eye_closure_percentage']:.1f}%\n")
                    f.write(f"  Longest Closure:        {metrics['longest_closure_frames']} frames\n")

                # Pose and Body Tracking Raw Values
                if self.last_pose_info is not None:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"POSE AND BODY TRACKING RAW VALUES:\n")
                    f.write(f"{'='*60}\n")
                    
                    # Body posture metrics
                    posture = self.last_pose_info['posture_metrics']

                    score = posture['overall_posture_score']
                    if score > 0.68:
                        quality_text = "Excellent"
                    elif score > 0.48:
                        quality_text = "Good"
                    else:
                        quality_text = "Needs Improvement"
                
                    f.write(f"\nBODY POSTURE ANALYSIS:\n")
                    f.write(f"  Trunk Inclination:\n")
                    f.write(f"    Forward/Backward: {posture['trunk_inclination_fb']:+.2f}° ({'Forward' if posture['trunk_inclination_fb'] > 0 else 'Backward' if posture['trunk_inclination_fb'] < 0 else 'Neutral'})\n")
                    f.write(f"    Left/Right Lean:  {posture['trunk_inclination_lr']:+.2f}° ({'Right' if posture['trunk_inclination_lr'] > 0 else 'Left' if posture['trunk_inclination_lr'] < 0 else 'Neutral'})\n")
                    
                    f.write(f"\n  Shoulder Symmetry:\n")
                    f.write(f"    Height Difference: {posture['shoulder_asymmetry']:+.2f}° ({'Right Higher' if posture['shoulder_asymmetry'] > 0 else 'Left Higher' if posture['shoulder_asymmetry'] < 0 else 'Level'})\n")
                    f.write(f"    Symmetry Score:    {posture['shoulder_symmetry_score']:.3f} (1.0=perfect)\n")
                    f.write(f"    Symmetry Status:    {posture['shoulder_symmetry_status']} {'✓' if posture['shoulder_symmetry_status'] == 'Correct' else '✗'}\n")

                    f.write(f"\n  Head Orientation:\n")
                    f.write(f"    Head Tilt:         {posture['head_tilt']:+.2f}° ({'Right' if posture['head_tilt'] > 0 else 'Left' if posture['head_tilt'] < 0 else 'Neutral'})\n")
                    f.write(f"    Head Turn:         {posture['head_turn']:+.2f}° ({'Right' if posture['head_turn'] > 0 else 'Left' if posture['head_turn'] < 0 else 'Forward'})\n")
                    
                    f.write(f"\n  Estimated Orientation:\n")
                    f.write(f"    Person is facing: {posture.get('orientation', 'Unknown')}\n")

                    f.write(f"\n  OVERALL POSTURE:\n")
                    f.write(f"    Overall Alignment: {posture['overall_posture_score']:.3f}\n")
                    f.write(f"    Posture Quality: {quality_text}\n")

                # ========== HAND TRACKING AND GESTURE ANALYSIS ==========
                if self.last_hand_info is not None:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"HAND TRACKING AND GESTURE ANALYSIS:\n")
                    f.write(f"{'='*60}\n")
                    
                    f.write(f"\nHAND DETECTION STATUS:\n")
                    f.write(f"  Hands Detected:         {'YES' if self.last_hand_info.get('hands_detected', False) else 'NO'}\n")
                    f.write(f"  Hands in Frame:         {'YES' if self.last_hand_info.get('hands_in_frame', False) else 'NO'}\n")
                    f.write(f"  Number of Hands:        {self.last_hand_info.get('num_hands', 0)}\n")
                    
                    # Individual hand analysis
                    hand_metrics = self.last_hand_info.get('hand_metrics', {})
                    if hand_metrics:
                        for hand_side, metrics in hand_metrics.items():
                            f.write(f"\n  {hand_side.upper()} HAND ANALYSIS:\n")
                            f.write(f"    Hand State:           {metrics.get('hand_state', 'unknown').upper()}\n")
                            f.write(f"    Openness:             {metrics.get('openness', 0):.3f} (0=closed, 1=open)\n")
                            f.write(f"    Finger Spread:        {metrics.get('finger_spread', 0):.3f} (0=together, 1=spread)\n")
                            f.write(f"    Primary Gesture:      {metrics.get('primary_gesture', 'none').upper()}\n")
                            f.write(f"    Gesture Confidence:   {metrics.get('gesture_confidence', 0):.3f}\n")
                            f.write(f"    Movement Velocity:    {metrics.get('velocity', 0):.4f}\n")
                            f.write(f"    Gesticulation Intensity: {metrics.get('gesticulation_intensity', 0):.3f}\n")
                            f.write(f"    Gesture Phase:        {metrics.get('gesture_phase', 'unknown').upper()}\n")
                            f.write(f"    Is Dominant Hand:     {'YES' if metrics.get('is_dominant', False) else 'NO'}\n")
                            
                            # Specific gestures detected
                            gestures = metrics.get('gestures_detected', {})
                            active_gestures = [g for g, detected in gestures.items() if detected]
                            if active_gestures:
                                f.write(f"    Active Gestures:      {', '.join(active_gestures).upper()}\n")
                            else:
                                f.write(f"    Active Gestures:      NONE\n")
                            
                            # Finger states
                            finger_states = metrics.get('finger_states', {})
                            if finger_states:
                                f.write(f"    Finger States:\n")
                                for finger_name, state in finger_states.items():
                                    curl = state.get('curl', 0)
                                    extended = state.get('extended', False)
                                    f.write(f"      {finger_name.capitalize():>6}: {'Extended' if extended else 'Curled'} (curl: {curl:.3f})\n")
                    
                    # Hand symmetry
                    if self.last_hand_info.get('hand_symmetry') is not None:
                        symmetry = self.last_hand_info['hand_symmetry']
                        f.write(f"\n  HAND SYMMETRY:\n")
                        f.write(f"    Symmetry Score:       {symmetry:.3f} (0=asymmetric, 1=symmetric)\n")
                        if symmetry > 0.7:
                            symmetry_status = "SYMMETRIC"
                        elif symmetry > 0.4:
                            symmetry_status = "MODERATELY SYMMETRIC"
                        else:
                            symmetry_status = "ASYMMETRIC"
                        f.write(f"    Symmetry Status:      {symmetry_status}\n")
                    
                    # Face interference detailed analysis
                    interference = self.last_hand_info.get('face_interference', {})
                    if interference:
                        f.write(f"\n  FACE INTERFERENCE ANALYSIS:\n")
                        f.write(f"    Currently Interfering:  {'YES' if interference.get('is_interfering', False) else 'NO'}\n")
                        f.write(f"    Interference Score:     {interference.get('interference_score', 0):.3f} (0=no overlap, 1=complete overlap)\n")
                        f.write(f"    Duration:               {interference.get('duration', 0):.1f} seconds\n")
                        f.write(f"    Sustained Interference: {'YES - FACE COVERED!' if interference.get('sustained_interference', False) else 'NO'}\n")
                        
                        interfering_hands = interference.get('interfering_hands', [])
                        if interfering_hands:
                            hand_names = []
                            for idx in interfering_hands:
                                if idx == 0 and 'left' in hand_metrics:
                                    hand_names.append('LEFT')
                                elif idx == 1 and 'right' in hand_metrics:
                                    hand_names.append('RIGHT')
                                else:
                                    hand_names.append(f'HAND_{idx}')
                            f.write(f"    Interfering Hands:      {', '.join(hand_names)}\n")
                        else:
                            f.write(f"    Interfering Hands:      NONE\n")
                else:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"HAND TRACKING: NO HAND DATA AVAILABLE\n")
                    f.write(f"{'='*60}\n")

                # Emotional quadrant
                quadrant = self._get_emotional_quadrant(valence, arousal)
                f.write(f"\nEmotional quadrant: {quadrant}\n")
                
                # Performance info
                f.write(f"\nPERFORMANCE:\n")
                f.write(f"Current FPS: {self.current_fps:.1f}\n")
                f.write(f"Device: {self.last_device}\n")
                
                f.write(f"{'='*60}\n")
                
            print(f"Debug TXT saved: {txt_filename}")
            
        except Exception as e:
            print(f"Failed to save debug TXT: {e}")


    def save_screenshot(self, frame: np.ndarray) -> str:
        """
        Save a screenshot of the current frame, corresponding debug JSON, and debug TXT.
        
        Args:
            frame: Frame to save
            
        Returns:
            Filename of saved screenshot
        """
        timestamp = int(time.time())
        filename = f"emotion_screenshot_{timestamp}.jpg"
        json_filename = f"emotion_snapshot_{timestamp}.json"
        txt_filename = f"emotion_debug_{timestamp}.txt"

        # Save image
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")
        
        # Save JSON
        self._save_debug_json(json_filename)
        
        # Save TXT with debug info
        self.save_debug_txt(txt_filename)
        
        return filename
    

    def _save_debug_json(self, json_filename: str) -> None:
        """Save the latest debug information as JSON."""

        def _convert_numpy_types(obj):
            """
            Recursively convert numpy types and tuples to standard Python types for JSON serialization.
            """
            if isinstance(obj, dict):
                return {k: _convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple, np.ndarray)):
                return [_convert_numpy_types(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif obj is None:
                return None
            else:
                return str(obj) if not isinstance(obj, (str, int, float, bool)) else obj

        # Enhanced hand info processing for JSON
        hand_info_processed = None
        if self.last_hand_info is not None:
            hand_info_processed = _convert_numpy_types(self.last_hand_info)
            
            # Add summary statistics for easier analysis
            hand_metrics = hand_info_processed.get('hand_metrics', {})
            if hand_metrics:
                summary = {
                    'total_hands': len(hand_metrics),
                    'hands_list': list(hand_metrics.keys()),
                    'any_hand_open': any(metrics.get('hand_state') == 'open' for metrics in hand_metrics.values()),
                    'any_hand_closed': any(metrics.get('hand_state') == 'closed' for metrics in hand_metrics.values()),
                    'dominant_hand': next((hand for hand, metrics in hand_metrics.items() if metrics.get('is_dominant')), None),
                    'active_gestures': []
                }
                
                # Collect all active gestures
                for hand_side, metrics in hand_metrics.items():
                    gestures = metrics.get('gestures_detected', {})
                    for gesture, active in gestures.items():
                        if active:
                            summary['active_gestures'].append(f"{hand_side}_{gesture}")
                
                hand_info_processed['hand_summary'] = summary

        data = {
            "timestamp": int(time.time()),
            "device": self.last_device,
            "face_bbox": self.last_face_bbox.tolist() if self.last_face_bbox is not None else None,
            "emotion_info": _convert_numpy_types(self.last_emotion_info) if self.last_emotion_info is not None else None,
            "iris_info": _convert_numpy_types(self.last_iris_info) if self.last_iris_info is not None else None,
            "pose_info": _convert_numpy_types(self.last_pose_info) if self.last_pose_info is not None else None,
            "distance_info": _convert_numpy_types(self.last_distance_info) if self.last_distance_info is not None else None,
            "hand_info": hand_info_processed
        }

        try:
            with open(json_filename, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Debug info saved: {json_filename}")
        except Exception as e:
            print(f"Failed to save JSON: {e}")