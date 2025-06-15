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
                        device: str) -> None:
        """
        Print detailed debug information to console.
        
        Args:
            face_bbox: Face bounding box coordinates
            emotion_info: Processed emotion information
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
        
        # Emotional quadrant
        quadrant = self._get_emotional_quadrant(valence, arousal)
        print(f"Emotional quadrant: {quadrant}")
        
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
                           emotion_info: Optional[Dict] = None) -> np.ndarray:
        """
        Create the final visualization with emotion information.
        
        Args:
            frame_bgr: Input frame in BGR format
            face_bbox: Face bounding box coordinates
            emotion_info: Processed emotion information
            
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
        
        # Draw FPS counter (only if option enabled)
        if self.show_fps:
            fps_text = f"FPS: {self.current_fps:.1f}"
            cv2.putText(vis_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Add debug indicator
        if self.debug:
            cv2.putText(vis_frame, "DEBUG MODE", (10, vis_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
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
        filename = f"emotion_screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")
        return filename