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
        self.last_device = "unknown"

    def set_last_debug_state(self,
                            face_bbox: Optional[np.ndarray],
                            emotion_info: Optional[Dict],
                            iris_info: Optional[Dict],
                            pose_info: Optional[Dict],
                            distance_info: Optional[Dict],
                            device: str) -> None:
        """Store the latest debug state for screenshot JSON logging."""
        self.last_face_bbox = face_bbox
        self.last_emotion_info = emotion_info
        self.last_iris_info = iris_info
        self.last_pose_info = pose_info
        self.last_distance_info = distance_info
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
                        device: str) -> None:
        """
        Print detailed debug information to console.
        
        Args:
            face_bbox: Face bounding box coordinates
            emotion_info: Processed emotion information
            iris_info: Raw iris tracking information
            pose_info: Pose tracking information
            distance_info: Distance measurement information
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
            
            print(f"\n  Head Orientation:")
            print(f"    Head Tilt:         {posture['head_tilt']:+.2f}° ({'Right' if posture['head_tilt'] > 0 else 'Left' if posture['head_tilt'] < 0 else 'Neutral'})")
            print(f"    Head Turn:         {posture['head_turn']:+.2f}° ({'Right' if posture['head_turn'] > 0 else 'Left' if posture['head_turn'] < 0 else 'Forward'})")
            
            print(f"\n  OVERALL POSTURE:")
            print(f"    Overall Alignment: {posture['overall_posture_score']:.3f}")

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
                           distance_info: Optional[Dict] = None) -> np.ndarray:
        """
        Create the final visualization with emotion, iris, and distance information.
        
        Args:
            frame_bgr: Input frame in BGR format
            face_bbox: Face bounding box coordinates
            emotion_info: Processed emotion information
            iris_info: Raw iris tracking information
            distance_info: Distance measurement information
            
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
            # Get status color
            color = distance_info['status_color']
            
            # Draw distance status bar at top of screen
            bar_height = 40
            bar_width = vis_frame.shape[1]
            overlay = vis_frame[0:bar_height, :].copy()
            cv2.rectangle(overlay, (0, 0), (bar_width, bar_height), color, -1)
            vis_frame[0:bar_height, :] = cv2.addWeighted(overlay, 0.3, vis_frame[0:bar_height, :], 0.7, 0)
            
            # Draw distance text
            distance_text = f"Distance: {distance_info['distance_cm']:.0f}cm - {distance_info['distance_status']}"
            cv2.putText(vis_frame, distance_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw iris information if available
        if iris_info is not None:
            # Draw eye aperture values on screen
            aperture_text = f"Eye Aperture - L: {iris_info['left_eye_aperture']:.3f} R: {iris_info['right_eye_aperture']:.3f}"
            cv2.putText(vis_frame, aperture_text, (10, vis_frame.shape[0] - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw iris centering value
            centering = iris_info['iris_position']['average_centering']
            centering_text = f"Iris Centering: {centering:.3f}"
            cv2.putText(vis_frame, centering_text, (10, vis_frame.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw blink count
            blink_text = f"Blinks: {iris_info['total_blinks']}"
            cv2.putText(vis_frame, blink_text, (10, vis_frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw FPS counter (only if option enabled)
        if self.show_fps:
            fps_text = f"FPS: {self.current_fps:.1f}"
            cv2.putText(vis_frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Add debug indicator
        if self.debug:
            cv2.putText(vis_frame, "DEBUG MODE", (10, vis_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return vis_frame
        
    def save_screenshot(self, frame: np.ndarray) -> str:
        """
        Save a screenshot of the current frame and corresponding debug JSON.
        
        Args:
            frame: Frame to save
            
        Returns:
            Filename of saved screenshot
        """
        timestamp = int(time.time())
        filename = f"emotion_screenshot_{timestamp}.jpg"
        json_filename = f"emotion_snapshot_{timestamp}.json"

        # Save image
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")
        
        # Save JSON
        self._save_debug_json(json_filename)
        
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


        data = {
            "timestamp": int(time.time()),
            "device": self.last_device,
            "face_bbox": self.last_face_bbox.tolist() if self.last_face_bbox is not None else None,
            "emotion_info": _convert_numpy_types(self.last_emotion_info) if self.last_emotion_info is not None else None,
            "iris_info": _convert_numpy_types(self.last_iris_info) if self.last_iris_info is not None else None,
            "pose_info": _convert_numpy_types(self.last_pose_info) if self.last_pose_info is not None else None,
            "distance_info": _convert_numpy_types(self.last_distance_info) if self.last_distance_info is not None else None
        }

        try:
            with open(json_filename, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Debug info saved: {json_filename}")
        except Exception as e:
            print(f"Failed to save JSON: {e}")