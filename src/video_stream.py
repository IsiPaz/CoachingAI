import cv2
import numpy as np
import threading
import queue
import time
from typing import Optional, Tuple, Callable

from emonet_handler import EmoNetHandler
from emotion_logger import EmotionLogger


class VideoStream:
    """
    Handles video capture, threading, and real-time processing coordination.
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
        
        # Thread-safe queues for frame processing
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        
        # Threading control
        self.processing_thread = None
        self.running = False
        
        # Camera setup
        self._setup_camera()
        
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
        
    def _processing_loop(self) -> None:
        """Main processing loop running in separate thread."""
        while self.running:
            try:
                # Get frame from queue (non-blocking)
                frame_bgr = self.frame_queue.get(timeout=0.1)
                
                # Process frame for emotion recognition
                face_bbox, emotion_info = self.emonet_handler.process_frame(frame_bgr)
                
                # Put result in queue
                if not self.result_queue.full():
                    self.result_queue.put((frame_bgr, face_bbox, emotion_info))
                    
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
                
                # Increment frame counter for logging
                self.logger.increment_frame_counter()
                
                # Add frame to processing queue (non-blocking)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                
                # Get processed result
                display_frame = frame
                face_bbox = None
                emotion_info = None
                
                try:
                    processed_frame, face_bbox, emotion_info = self.result_queue.get_nowait()
                    display_frame = processed_frame
                except queue.Empty:
                    # Use original frame
                    pass
                
                # Print debug info
                self.logger.print_debug_info(face_bbox, emotion_info, self.emonet_handler.device)
                
                # Create visualization
                display_frame = self.logger.create_visualization(display_frame, face_bbox, emotion_info)
                
                # Display frame
                cv2.imshow('Real-time Emotion Recognition', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    self.logger.save_screenshot(display_frame)
                elif key == ord('d') and self.logger.debug:
                    # Toggle debug print interval
                    self.logger.toggle_debug_interval()
                
                # Update FPS
                self.logger.update_fps()
                
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