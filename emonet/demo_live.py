import cv2
import numpy as np
import torch
from torch import nn
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import threading
import queue
import time

from face_alignment.detection.sfd.sfd_detector import SFDDetector
from emonet.models import EmoNet


class RealtimeEmotionRecognizer:
    """
    Real-time emotion recognition system using EmoNet with GPU optimization.
    Implements multi-threading for optimal performance with live camera feed.
    """
    
    def __init__(self, 
                 n_expression: int = 8,
                 device: str = "cuda:0",
                 image_size: int = 256,
                 camera_id: int = 0,
                 target_fps: int = 30,
                 show_fps: bool = False,
                 show_circumplex: bool = False,
                 debug: bool = False):
        """
        Initialize the real-time emotion recognition system.
        
        Args:
            n_expression: Number of emotion classes (5 or 8)
            device: Device to run models on (cuda:0, cpu)
            image_size: Size for emotion recognition input
            camera_id: Camera device ID
            target_fps: Target FPS for processing
            show_fps: Whether to display FPS counter (debug option)
            show_circumplex: Whether to display circumplex visualization (debug option)
        """
        self.device = device
        self.image_size = image_size
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.show_fps = show_fps
        self.show_circumplex = show_circumplex
        self.debug = debug
        
        # Emotion class mapping
        self.emotion_classes = {
            0: "Neutral",
            1: "Happy", 
            2: "Sad",
            3: "Surprise",
            4: "Fear",
            5: "Disgust",
            6: "Anger",
            7: "Contempt"
        }
        
        # Thread-safe queues for frame processing
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        
        # Threading control
        self.processing_thread = None
        self.running = False
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0

        # Debug counters
        self.debug_frame_counter = 0
        self.debug_print_interval = 30  
        
        # Initialize models
        self._load_models(n_expression)
        self._setup_camera()
        
    def _load_models(self, n_expression: int) -> None:
        """Load EmoNet and face detection models with GPU optimization."""
        print(f"Loading models on device: {self.device}")
        
        # Load EmoNet
        state_dict_path = Path(__file__).parent.joinpath(
            "pretrained", f"emonet_{n_expression}.pth"
        )
        
        if not state_dict_path.exists():
            raise FileNotFoundError(f"EmoNet model not found at {state_dict_path}")
            
        print(f"Loading EmoNet model from {state_dict_path}")
        state_dict = torch.load(str(state_dict_path), map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        self.emonet = EmoNet(n_expression=n_expression).to(self.device)
        self.emonet.load_state_dict(state_dict, strict=False)
        self.emonet.eval()
        
        # Enable GPU optimizations
        if self.device.startswith('cuda'):
            torch.backends.cudnn.benchmark = True
            # Warm up GPU
            self._warmup_gpu()
        
        # Load face detector
        print("Loading SFD face detector")
        self.face_detector = SFDDetector(self.device)
        
        print("Models loaded successfully")
        
    def _warmup_gpu(self) -> None:
        """Warm up GPU with dummy inference for optimal performance."""
        print("Warming up GPU...")
        dummy_input = torch.randn(1, 3, self.image_size, self.image_size).to(self.device)
        
        with torch.no_grad():
            for _ in range(5):
                _ = self.emonet(dummy_input)
        
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()
        print("GPU warmup completed")
        
    def _setup_camera(self) -> None:
        """Initialize camera with optimal settings."""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
            
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        # Get actual camera resolution
        self.camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Camera initialized: {self.camera_width}x{self.camera_height}")
        
    def _detect_faces(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
        """
        Detect faces in the frame using SFD detector.
        
        Args:
            frame_bgr: Input frame in BGR format
            
        Returns:
            List of detected face bounding boxes
        """
        with torch.no_grad():
            detected_faces = self.face_detector.detect_from_image(frame_bgr)
        return detected_faces
        
    def _run_emotion_recognition(self, face_crop_rgb: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Run emotion recognition on a face crop.
        
        Args:
            face_crop_rgb: Face crop in RGB format
            
        Returns:
            Dictionary containing emotion predictions
        """
        # Resize face crop to model input size
        face_resized = cv2.resize(face_crop_rgb, (self.image_size, self.image_size))
        
        # Convert to tensor and normalize
        face_tensor = torch.tensor(face_resized).permute(2, 0, 1).to(self.device) / 255.0
        
        with torch.no_grad():
            emotion_output = self.emonet(face_tensor.unsqueeze(0))
            
        return emotion_output
        
    def _print_debug_info(self, face_bbox: np.ndarray, emotion_result: Dict[str, torch.Tensor]) -> None:
        """Print detailed debug information to console."""
        if not self.debug:
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
            
        if emotion_result is None:
            print("No emotion analysis available")
            return
            
        # Emotion probabilities
        emotion_probs = nn.functional.softmax(emotion_result["expression"], dim=1)
        predicted_class = torch.argmax(emotion_probs).cpu().item()
        confidence = emotion_probs[0, predicted_class].cpu().item()
        
        print(f"\nEMOTION ANALYSIS:")
        print(f"Predicted emotion: {self.emotion_classes[predicted_class]} (confidence: {confidence:.3f})")
        
        print(f"\nAll emotion probabilities:")
        for i, prob in enumerate(emotion_probs[0].cpu().numpy()):
            print(f"  {self.emotion_classes[i]:>10}: {prob:.3f} {'<-- PREDICTED' if i == predicted_class else ''}")
            
        # Valence and Arousal
        valence = emotion_result["valence"].clamp(-1.0, 1.0).cpu().item()
        arousal = emotion_result["arousal"].clamp(-1.0, 1.0).cpu().item()
        
        print(f"\nVALENCE-AROUSAL ANALYSIS:")
        print(f"Valence: {valence:+.3f} ({'Positive' if valence > 0 else 'Negative' if valence < 0 else 'Neutral'})")
        print(f"Arousal:  {arousal:+.3f} ({'High' if arousal > 0 else 'Low' if arousal < 0 else 'Neutral'})")
        
        # Emotional quadrant
        if valence > 0 and arousal > 0:
            quadrant = "High Arousal + Positive Valence (Excited/Happy)"
        elif valence > 0 and arousal < 0:
            quadrant = "Low Arousal + Positive Valence (Calm/Content)"
        elif valence < 0 and arousal > 0:
            quadrant = "High Arousal + Negative Valence (Stressed/Angry)"
        elif valence < 0 and arousal < 0:
            quadrant = "Low Arousal + Negative Valence (Sad/Depressed)"
        else:
            quadrant = "Neutral state"
            
        print(f"Emotional quadrant: {quadrant}")
        
        # Performance info
        print(f"\nPERFORMANCE:")
        print(f"Current FPS: {self.current_fps:.1f}")
        print(f"Target FPS: {self.target_fps}")
        print(f"Device: {self.device}")
        
        print(f"{'='*60}\n")

    def _create_circumplex_visualization(self, valence: float, arousal: float) -> np.ndarray:
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
        
    def _create_visualization(self, 
                            frame_bgr: np.ndarray,
                            face_bbox: Optional[np.ndarray] = None,
                            emotion_result: Optional[Dict[str, torch.Tensor]] = None) -> np.ndarray:
        """
        Create the final visualization with emotion information.
        
        Args:
            frame_bgr: Input frame in BGR format
            face_bbox: Face bounding box coordinates
            emotion_result: Emotion recognition results
            
        Returns:
            Visualization frame
        """
        vis_frame = frame_bgr.copy()
        
        if face_bbox is not None and emotion_result is not None:
            # Draw face bounding box
            cv2.rectangle(vis_frame, 
                         (face_bbox[0], face_bbox[1]), 
                         (face_bbox[2], face_bbox[3]), 
                         (0, 255, 0), 3)
            
            # Get predicted emotion
            emotion_probs = nn.functional.softmax(emotion_result["expression"], dim=1)
            predicted_class = torch.argmax(emotion_probs).cpu().item()
            confidence = emotion_probs[0, predicted_class].cpu().item()
            
            emotion_text = f"{self.emotion_classes[predicted_class]}: {confidence:.2f}"
            
            # Draw emotion text
            cv2.putText(vis_frame, emotion_text,
                       (face_bbox[0], face_bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Get valence and arousal
            valence = emotion_result["valence"].clamp(-1.0, 1.0).cpu().item()
            arousal = emotion_result["arousal"].clamp(-1.0, 1.0).cpu().item()
            
            # Draw valence/arousal info
            va_text = f"V: {valence:.2f}, A: {arousal:.2f}"
            cv2.putText(vis_frame, va_text,
                       (face_bbox[0], face_bbox[3] + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Print debug info to console
            self._print_debug_info(face_bbox, emotion_result)

            # Create circumplex visualization (only if debug option enabled)
            if self.show_circumplex:
                circumplex = self._create_circumplex_visualization(valence, arousal)
                
                # Overlay circumplex on frame (top-right corner)
                overlay_size = min(200, vis_frame.shape[1]//4)
                circumplex_resized = cv2.resize(circumplex, (overlay_size, overlay_size))
                
                y_offset = 10
                x_offset = vis_frame.shape[1] - overlay_size - 10
                
                # Add semi-transparent overlay
                overlay_region = vis_frame[y_offset:y_offset+overlay_size, x_offset:x_offset+overlay_size]
                blended = cv2.addWeighted(overlay_region, 0.3, circumplex_resized, 0.7, 0)
                vis_frame[y_offset:y_offset+overlay_size, x_offset:x_offset+overlay_size] = blended
        
        else:
            # Print debug info even when no face is detected
            if self.debug and self.debug_frame_counter % self.debug_print_interval == 0:
                print(f"\nDEBUG - Frame {self.debug_frame_counter}: No face detected")
        
        # Draw FPS counter (only if debug option enabled)
        if self.show_fps:
            fps_text = f"FPS: {self.current_fps:.1f}"
            cv2.putText(vis_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Add debug indicator
        if self.debug:
            cv2.putText(vis_frame, "DEBUG MODE", (10, vis_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return vis_frame
        
    def _processing_loop(self) -> None:
        """Main processing loop running in separate thread."""
        while self.running:
            try:
                # Get frame from queue (non-blocking)
                frame_bgr = self.frame_queue.get(timeout=0.1)
                
                # Convert to RGB for face detection
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                detected_faces = self._detect_faces(frame_bgr)
                
                face_bbox = None
                emotion_result = None
                
                if len(detected_faces) > 0:
                    # Use first detected face
                    face_bbox = np.array(detected_faces[0]).astype(np.int32)
                    
                    # Extract face crop
                    face_crop_rgb = frame_rgb[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
                    
                    if face_crop_rgb.size > 0:
                        # Run emotion recognition
                        emotion_result = self._run_emotion_recognition(face_crop_rgb)
                
                # Put result in queue
                if not self.result_queue.full():
                    self.result_queue.put((frame_bgr, face_bbox, emotion_result))
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                
    def _update_fps(self) -> None:
        """Update FPS counter (only if debug option enabled)."""
        if self.show_fps:
            self.fps_counter += 1
            if self.fps_counter >= 30:
                current_time = time.time()
                self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
                self.fps_counter = 0
                self.fps_start_time = current_time
            
    def start(self) -> None:
        """Start the real-time emotion recognition system."""
        print("Starting real-time emotion recognition...")
        print("Press 'q' to quit, 's' to save screenshot")
        if self.debug:
            print("DEBUG MODE ENABLED - Detailed console output every 30 frames")
        
        self.running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Main display loop
        last_frame_time = time.time()
        
        try:
            while self.running:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Increment debug frame counter
                self.debug_frame_counter += 1
                
                # Add frame to processing queue (non-blocking)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                
                # Get processed result
                display_frame = frame
                try:
                    processed_frame, face_bbox, emotion_result = self.result_queue.get_nowait()
                    display_frame = self._create_visualization(processed_frame, face_bbox, emotion_result)
                except queue.Empty:
                    # Use original frame with FPS counter
                    display_frame = self._create_visualization(frame)
                
                # Display frame
                cv2.imshow('Real-time Emotion Recognition', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = int(time.time())
                    filename = f"emotion_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"Screenshot saved: {filename}")
                elif key == ord('d') and self.debug:
                    # Toggle debug print interval
                    self.debug_print_interval = 1 if self.debug_print_interval == 30 else 30
                    print(f"Debug print interval changed to: {self.debug_print_interval} frames")
                
                # Update FPS
                self._update_fps()
                
                # Control frame rate
                current_time = time.time()
                elapsed = current_time - last_frame_time
                sleep_time = max(0, self.frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_frame_time = time.time()
                
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.stop()
            
    def stop(self) -> None:
        """Stop the real-time emotion recognition system."""
        print("Stopping emotion recognition...")
        self.running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
            
        if self.cap:
            self.cap.release()
            
        cv2.destroyAllWindows()
        print("Stopped successfully")


def main():
    """Main function to run the real-time emotion recognition system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time Emotion Recognition with EmoNet")
    parser.add_argument("--nclasses", type=int, default=8, choices=[5, 8],
                       help="Number of emotion classes (5 or 8)")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to run on (cuda:0, cpu)")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera device ID")
    parser.add_argument("--fps", type=int, default=30,
                       help="Target FPS")
    parser.add_argument("--show-fps", action="store_true", default=False,
                       help="Show FPS counter (debug option)")
    parser.add_argument("--show-circumplex", action="store_true", default=False,
                       help="Show valence-arousal circumplex visualization (debug option)")
    parser.add_argument("--debug", action="store_true", default=False,
                       help="Enable all debug visualizations (FPS + circumplex)")
    
    args = parser.parse_args()
    
    # Handle debug options
    show_fps = args.show_fps or args.debug
    show_circumplex = args.show_circumplex or args.debug
    
    # Check CUDA availability
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = "cpu"
    
    try:
        # Create and start the emotion recognition system
        recognizer = RealtimeEmotionRecognizer(
            n_expression=args.nclasses,
            device=args.device,
            camera_id=args.camera,
            target_fps=args.fps,
            show_fps=show_fps,
            show_circumplex=show_circumplex,
            debug=args.debug
        )
        
        recognizer.start()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the required dependencies and pretrained models")


if __name__ == "__main__":
    main()