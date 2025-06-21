import cv2
import numpy as np
import threading
import queue
import time
from typing import Optional, Tuple, Callable
from pose_handler import PoseHandler
from emonet_handler import EmoNetHandler
from emotion_logger import EmotionLogger
from iris_handler import IrisHandler
from distance_handler import DistanceHandler
from hand_handler import HandHandler


class VideoStream:
    """
    Handles video capture, threading, and real-time processing coordination.
    Optimized for smooth frame processing and display.
    """
    
    def __init__(self,
                 emonet_handler: EmoNetHandler,
                 logger: EmotionLogger,
                 camera_id: int = 0,
                 target_fps: int = 30):
        """
        Initialize the video stream.
        
        Args:
            emonet_handler: EmoNet handler instance
            logger: Emotion logger instance
            camera_id: Camera device ID
            target_fps: Target FPS for processing
        """
        self.emonet_handler = emonet_handler
        self.logger = logger
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        
        # Initialize handlers with same device as emonet
        self.iris_handler = IrisHandler(device=emonet_handler.device)
        self.pose_handler = PoseHandler(device=emonet_handler.device)
        self.distance_handler = DistanceHandler(device=emonet_handler.device)
        self.hand_handler = HandHandler(device=emonet_handler.device)
        
        # Optimized queues - smaller buffers for real-time processing
        self.frame_queue = queue.Queue(maxsize=1)  # Reduced buffer
        self.result_queue = queue.Queue(maxsize=1)  # Reduced buffer
        
        # Threading control
        self.processing_thread = None
        self.running = False
        
        # Processing optimization
        self.skip_frames = 0  # Skip frames for processing
        self.processing_interval = 2  # Process every N frames for heavy operations
        self.frame_counter = 0
        
        # Last valid results for smooth display
        self.last_results = {
            'face_bbox': None,
            'emotion_info': None,
            'iris_info': None,
            'pose_info': None,
            'distance_info': None,
            'hand_info': None
        }
        
        # Camera setup
        self._setup_camera()
        
    def _setup_camera(self) -> None:
        """Initialize camera with optimal settings for smooth capture."""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
            
        # Optimized camera properties
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        # Set resolution for better performance (optional)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Get actual camera resolution
        self.camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Camera initialized: {self.camera_width}x{self.camera_height}")
        
    def _processing_loop(self) -> None:
        """Optimized processing loop with frame skipping for heavy operations."""
        local_frame_counter = 0
        
        while self.running:
            try:
                # Get frame from queue (non-blocking with short timeout)
                frame_bgr = self.frame_queue.get(timeout=0.05)
                local_frame_counter += 1
                
                # Convert to RGB once for all MediaPipe operations
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                
                # Always process emotion (fastest operation)
                face_bbox, emotion_info = self.emonet_handler.process_frame(frame_bgr)
                
                # Process iris and pose only every N frames to reduce load
                iris_info = None
                pose_info = None
                distance_info = None
                hand_info = None
                
                if face_bbox is not None:
                    # Process iris every frame (lightweight)
                    iris_info = self.iris_handler.process_frame(frame_rgb)
                    
                    # Process distance every frame (lightweight)
                    distance_info = self.distance_handler.process_frame(
                        face_bbox, iris_info, frame_bgr.shape[:2]
                    )
                    
                    # Process pose less frequently (heavier operation)
                    if local_frame_counter % self.processing_interval == 0:
                        pose_info = self.pose_handler.process_frame(frame_rgb)
                    else:
                        # Use last pose result for smoother display
                        pose_info = self.last_results.get('pose_info')
                
                # Process hands every frame (important for gesture recognition)
                # Pass face bbox for interference detection
                hand_info = self.hand_handler.process_frame(frame_rgb, face_bbox)
                
                # Update last results
                if face_bbox is not None:
                    self.last_results['face_bbox'] = face_bbox
                if emotion_info is not None:
                    self.last_results['emotion_info'] = emotion_info
                if iris_info is not None:
                    self.last_results['iris_info'] = iris_info
                if pose_info is not None:
                    self.last_results['pose_info'] = pose_info
                if distance_info is not None:
                    self.last_results['distance_info'] = distance_info
                if hand_info is not None:
                    self.last_results['hand_info'] = hand_info
                
                # Put result in queue (drop old frames for real-time processing)
                try:
                    self.result_queue.put_nowait((
                        frame_bgr, 
                        self.last_results['face_bbox'], 
                        self.last_results['emotion_info'], 
                        self.last_results['iris_info'], 
                        self.last_results['pose_info'],
                        self.last_results['distance_info'],
                        self.last_results['hand_info']
                    ))
                except queue.Full:
                    # Drop old result and put new one
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait((
                            frame_bgr, 
                            self.last_results['face_bbox'], 
                            self.last_results['emotion_info'], 
                            self.last_results['iris_info'], 
                            self.last_results['pose_info'],
                            self.last_results['distance_info'],
                            self.last_results['hand_info']
                        ))
                    except queue.Empty:
                        pass
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                if self.logger.debug:
                    import traceback
                    traceback.print_exc()
                    
    def start(self) -> None:
        """Start the video stream and processing."""
        print("Starting real-time emotion recognition...")
        print("Press 'q' to quit, 's' to save screenshot")
        if self.logger.debug:
            print("DEBUG MODE ENABLED - Detailed console output")
            print("Press 'd' to toggle debug print interval")
            print("Press '+' to increase processing interval, '-' to decrease")
        
        self.running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Main display loop - optimized for smooth display
        last_frame_time = time.time()
        display_frame_time = 1.0 / 60  # Display at 60fps for smoothness
        
        try:
            while self.running:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                self.frame_counter += 1
                self.logger.increment_frame_counter()
                
                # Add frame to processing queue (drop frames if queue is full)
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # Drop old frame and put new one
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
                
                # Get processed result
                display_frame = frame
                face_bbox = None
                emotion_info = None
                iris_info = None
                pose_info = None
                distance_info = None
                hand_info = None
                
                try:
                    processed_frame, face_bbox, emotion_info, iris_info, pose_info, distance_info, hand_info = self.result_queue.get_nowait()
                    display_frame = processed_frame
                except queue.Empty:
                    # Use original frame with last known results for smooth display
                    face_bbox = self.last_results.get('face_bbox')
                    emotion_info = self.last_results.get('emotion_info')
                    iris_info = self.last_results.get('iris_info')
                    pose_info = self.last_results.get('pose_info')
                    distance_info = self.last_results.get('distance_info')
                    hand_info = self.last_results.get('hand_info')
                
                # Print debug info only occasionally to avoid performance hit
                if self.logger.debug and self.frame_counter % 10 == 0:
                    self.logger.print_debug_info(face_bbox, emotion_info, iris_info, pose_info, distance_info, hand_info, self.emonet_handler.device)
                
                # Apply visualizations
                if iris_info is not None:
                    display_frame = self.iris_handler.draw_iris_visualization(display_frame, iris_info, self.logger.debug)
                
                if pose_info is not None:
                    display_frame = self.pose_handler.draw_pose_visualization(display_frame, pose_info, self.logger.debug)
                
                if hand_info is not None:
                    display_frame = self.hand_handler.draw_hand_visualization(display_frame, hand_info, self.logger.debug)
                            
                # Create emotion visualization (lightweight)
                display_frame = self.logger.create_visualization(display_frame, face_bbox, emotion_info, iris_info, distance_info, pose_info, hand_info)

                
                # Display frame
                cv2.imshow('Real-time Emotion Recognition', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.logger.set_last_debug_state(face_bbox, emotion_info, iris_info, pose_info, distance_info, hand_info, self.emonet_handler.device)
                    self.logger.save_screenshot(display_frame)
                elif key == ord('d') and self.logger.debug:
                    self.logger.toggle_debug_interval()
                elif key == ord('+') or key == ord('='):
                    # Increase processing interval (reduce load)
                    self.processing_interval = min(10, self.processing_interval + 1)
                    print(f"Processing interval increased to: {self.processing_interval}")
                elif key == ord('-'):
                    # Decrease processing interval (increase accuracy)
                    self.processing_interval = max(1, self.processing_interval - 1)
                    print(f"Processing interval decreased to: {self.processing_interval}")
                
                # Update FPS counter less frequently
                if self.frame_counter % 5 == 0:
                    self.logger.update_fps()
                
                # Smooth frame rate control
                current_time = time.time()
                elapsed = current_time - last_frame_time
                sleep_time = max(0, display_frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_frame_time = time.time()
                
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.stop()
            
    def stop(self) -> None:
        """Stop the video stream and processing."""
        print("Stopping emotion recognition...")
        self.running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
            
        if self.cap:
            self.cap.release()
            
        cv2.destroyAllWindows()
        print("Stopped successfully")
        
    def is_running(self) -> bool:
        """Check if the video stream is running."""
        return self.running
        
    def get_camera_info(self) -> Tuple[int, int]:
        """Get camera resolution."""
        return self.camera_width, self.camera_height