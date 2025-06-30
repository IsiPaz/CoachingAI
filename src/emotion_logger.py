import os
import cv2
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
            print(f"\n IRIS POSITION (normalized -1 to +1):")
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
        
        # Draw face detection results
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
        if distance_info is not None and isinstance(distance_info, dict):
            distance_cm = distance_info.get('distance_cm', 0)
            distance_status = distance_info.get('distance_status', 'Unknown')
            status_color = distance_info.get('status_color', (255, 255, 255))
            
            distance_text = f"Distance: {distance_cm:.0f}cm - {distance_status}"
            font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            (text_w, text_h), _ = cv2.getTextSize(distance_text, font, scale, thickness)
            x, y = 10, 25

            # Draw white background rectangle
            cv2.rectangle(vis_frame, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), (255, 255, 255), -1)
            # Draw colored text on top
            cv2.putText(vis_frame, distance_text, (x, y), font, scale, status_color, thickness)
                    
        # Draw iris information if available - FIXED SECTION
        if iris_info is not None and isinstance(iris_info, dict):           
            # Draw blink count with safety check
            total_blinks = iris_info.get('total_blinks', 0)
            blink_text = f"Blinks: {total_blinks}"
            f, s, t = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            (w, h), _ = cv2.getTextSize(blink_text, f, s, t)
            org = (10, 70 + h)
            cv2.rectangle(vis_frame, (org[0]-5, org[1]-h-5), (org[0]+w+5, org[1]+5), (255,255,255), -1)
            cv2.putText(vis_frame, blink_text, org, f, s, (0,0,0), t)
                    
        # Draw hand interference warning if sustained
        x, y = 10, vis_frame.shape[0] - 307 + 18

        if hand_info and isinstance(hand_info, dict):
            face_interference = hand_info.get('face_interference', {})
            if isinstance(face_interference, dict) and face_interference.get('sustained_interference', False):
                duration = face_interference.get('duration', 0.0)
                alert_text = "FACE COVERED"
                duration_text = f"Duration: {duration:.1f}s"
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

        # Draw pose information in debug mode - FIXED SECTION
        if self.debug and pose_info is not None and isinstance(pose_info, dict):
            posture_metrics = pose_info.get('posture_metrics', {})
            if isinstance(posture_metrics, dict):
                orientation = posture_metrics.get("orientation", "Unknown")
                f, s, t = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                
                # Calculate position safely
                if iris_info is not None and isinstance(iris_info, dict):
                    total_blinks = iris_info.get('total_blinks', 0)
                    blink_h = cv2.getTextSize(f"Blinks: {total_blinks}", f, s, t)[0][1]
                else:
                    blink_h = cv2.getTextSize("Blinks: 0", f, s, t)[0][1]
                    
                org = (10, 70 + blink_h + 15 + blink_h) 

                # Force color: green if frontal, else red (any other case)
                color = (0, 255, 0) if orientation.lower() == "frontal" else (0, 0, 255)

                (w, h), _ = cv2.getTextSize(f"Orientation: {orientation}", f, s, t)
                cv2.rectangle(vis_frame, (org[0]-5, org[1]-h-5), (org[0]+w+5, org[1]+5), (255,255,255), -1)
                cv2.putText(vis_frame, f"Orientation: {orientation}", org, f, s, color, t)

                # Draw shoulder symmetry status
                shoulder_status = posture_metrics.get('shoulder_symmetry_status', 'Unknown')
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

        # Position calculation for additional info - FIXED
        if self.debug and iris_info is not None and isinstance(iris_info, dict):
            total_blinks = iris_info.get('total_blinks', 0)
            blink_h = cv2.getTextSize(f"Blinks: {total_blinks}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][1]
            y_pos = 100 + 15 + blink_h + 17 + blink_h + 35  # Below orientation
        else:
            y_pos = 150  # Fixed position
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2

        # Add debug mode indicator
        if self.debug:
            text = "DEBUG MODE"
            f, s, t = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            (w, h), _ = cv2.getTextSize(text, f, s, t)
            org = (vis_frame.shape[1] - w - 10, 10 + h)
            cv2.rectangle(vis_frame, (org[0]-5, org[1]-h-5), (org[0]+w+5, org[1]+5), (255,255,255), -1)
            cv2.putText(vis_frame, text, org, f, s, (255,0,0), t)
        
        return vis_frame
        

    def save_screenshot(self, frame: np.ndarray) -> str:
        """
        Save a screenshot of the current frame.
        
        Args:
            frame: Frame to save
            
        Returns:
            Filename of saved screenshot
        """
        timestamp = int(time.time())
        os.makedirs('screenshots', exist_ok=True)

        filename = f"screenshots/emotion_screenshot_{timestamp}.jpg"

        # Save image only
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")
        
        return filename