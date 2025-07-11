import cv2
import numpy as np
import threading
import time
from typing import Optional, Tuple, Dict, Any
from collections import deque
import torch
from pose_handler import PoseHandler
from emonet_handler import EmoNetHandler
from emotion_logger import EmotionLogger
from iris_handler import IrisHandler
from distance_handler import DistanceHandler
from hand_handler import HandHandler
from feedback_handler import FeedbackHandler
from report_handler import ReportHandler


class VideoStream:
    """
    GPU-optimized video stream handler for real-time emotion recognition.
    Designed for stable 10 FPS processing with all features enabled.
    """
    
    def __init__(self,
                 emonet_handler: EmoNetHandler,
                 logger: EmotionLogger,
                 camera_id: int = 0,
                 target_fps: int = 10,
                 show_fps: bool = False,
                 openai_api_key: Optional[str] = None):
        """
        Initialize GPU-optimized video stream for 10 FPS target.
        
        Args:
            emonet_handler: EmoNet handler instance
            logger: Emotion logger instance
            camera_id: Camera device ID
            target_fps: Target FPS (default: 10)
            openai_api_key: Optional OpenAI API key for feedback
        """
        self.emonet_handler = emonet_handler
        self.logger = logger
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps  # 100ms per frame at 10 FPS
        self.show_fps = show_fps
        
        # Device configuration
        self.device = emonet_handler.device
        self.gpu_available = torch.cuda.is_available() and str(self.device) != 'cpu'
        
        # GPU optimization
        if self.gpu_available:
            torch.cuda.set_device(self.device)
            torch.backends.cudnn.benchmark = True
            # Set CUDA to use less memory fragmentation
            torch.cuda.empty_cache()
            print(f"GPU optimization enabled on {self.device}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(self.device).total_memory / 1e9:.1f} GB")
        
        # Initialize all handlers on same device
        self.iris_handler = IrisHandler(device=self.device)
        self.pose_handler = PoseHandler(device=self.device)
        self.distance_handler = DistanceHandler(device=self.device)
        self.hand_handler = HandHandler(device=self.device)
        
        # Initialize feedback handler
        self._init_feedback_handler(openai_api_key)
        
        # Initialize report handler
        self.report_handler = ReportHandler(
            openai_client=getattr(self.feedback_handler, 'client', None) if self.feedback_handler else None,
            sample_interval=2.0  # Collect metrics every 2 seconds
        )
        
        # Double buffering para frames
        self.current_frame = None
        self.previous_frame = None
        self.frame_lock = threading.Lock()
        
        # buffer reutilizable
        self.rgb_buffer = None
        
        # Results storage
        self.latest_results = {
            'face_bbox': None,
            'emotion_info': None,
            'iris_info': None,
            'pose_info': None,
            'distance_info': None,
            'hand_info': None
        }
        self.results_lock = threading.Lock()
        
        # Threading
        self.capture_thread = None
        self.process_thread = None
        self.running = False
        
        # Performance tracking
        self.frame_counter = 0
        self.fps_tracker = {'capture': 0, 'process': 0, 'display': 0}
        self.last_fps_update = time.perf_counter()
        self.process_times = deque(maxlen=30)
        
        # Display 
        self.display_frame = None
        
        # Camera setup
        self._setup_camera()
        
    def _init_feedback_handler(self, api_key: Optional[str]) -> None:
        """Initialize feedback handler with error handling."""
        self.feedback_handler = None
        if api_key:
            try:
                self.feedback_handler = FeedbackHandler(
                    api_key=api_key,
                    model="gpt-4o-mini",
                    feedback_interval=15.0
                )
                self.feedback_handler.start()
                print("ChatGPT feedback enabled")
            except Exception as e:
                print(f"Failed to initialize feedback handler: {e}")
                
    def _setup_camera(self) -> None:
        """Initialize camera with optimal settings for 10 FPS."""
        # Try backend in order of performance
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2]
        
        for backend in backends:
            try:
                self.cap = cv2.VideoCapture(self.camera_id, backend)
                if self.cap.isOpened():
                    print(f"Camera opened with backend: {backend}")
                    break
            except:
                continue
        else:
            # Fallback to default
            self.cap = cv2.VideoCapture(self.camera_id)
            
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
        
        # Optimize camera settings
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffering
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        # Try to use MJPEG for faster decoding
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Get actual properties
        self.camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera initialized: {self.camera_width}x{self.camera_height} @ {actual_fps:.1f} FPS")
        
        # Discard first frames to stabilize
        print("Stabilizing camera...")
        for _ in range(5):
            ret, _ = self.cap.read()
            if ret:
                time.sleep(0.1)
        
    def _capture_loop(self) -> None:
        """Dedicated thread for continuous frame capture."""
        frame_count = 0
        last_time = time.perf_counter()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Simple double buffering
            with self.frame_lock:
                if self.current_frame is not None:
                    self.previous_frame = self.current_frame.copy()
                self.current_frame = frame.copy()  # Siempre hacer copia
            
            # FPS tracking
            frame_count += 1
            if frame_count % 10 == 0:
                current_time = time.perf_counter()
                self.fps_tracker['capture'] = 10 / (current_time - last_time)
                last_time = current_time


    def _process_loop(self) -> None:
        """GPU-optimized processing thread."""
        frame_count = 0  # KEEP - used for FPS and GPU sync
        last_time = time.perf_counter()
        
        while self.running:
            # Get current frame safely
            frame = None
            with self.frame_lock:
                if self.current_frame is not None:
                    frame = self.current_frame.copy()
                    
            if frame is None:
                time.sleep(0.01)
                continue
                
            process_start = time.perf_counter()
            
            # Reuse RGB buffer
            if self.rgb_buffer is None or self.rgb_buffer.shape != frame.shape:
                self.rgb_buffer = np.empty_like(frame)
                
            # Fast color conversion
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self.rgb_buffer)
            
            # Process all components
            results = self._process_frame_gpu(frame, self.rgb_buffer)
            
            # Update results atomically
            with self.results_lock:
                # Check if person left the scene completely
                person_in_scene = (
                    results.get('face_bbox') is not None or 
                    (results.get('hand_info') and results['hand_info'].get('hands_detected', False))
                )
                
                if not person_in_scene:
                    # Clear ALL results when no person detected
                    self.latest_results = {
                        'face_bbox': None,
                        'emotion_info': None,
                        'iris_info': None,
                        'pose_info': None,
                        'distance_info': None,
                        'hand_info': None
                    }
                else:
                    # Update normally
                    for key, value in results.items():
                        self.latest_results[key] = value
                        
            # Send to feedback handler
            if self.feedback_handler:
                self.feedback_handler.analyze_state(
                    emotion_info=self.latest_results['emotion_info'],
                    iris_info=self.latest_results['iris_info'],
                    pose_info=self.latest_results['pose_info'],
                    distance_info=self.latest_results['distance_info'],
                    hand_info=self.latest_results['hand_info']
                )
            
            # Collect metrics for report
            self.report_handler.collect_metrics(
                emotion_info=self.latest_results['emotion_info'],
                iris_info=self.latest_results['iris_info'],
                pose_info=self.latest_results['pose_info'],
                distance_info=self.latest_results['distance_info'],
                hand_info=self.latest_results['hand_info']
            )
                
            # Performance tracking
            process_time = time.perf_counter() - process_start
            self.process_times.append(process_time)
            
            # IMPORTANT: KEEP frame_count and its logic
            frame_count += 1
            if frame_count % 10 == 0:
                current_time = time.perf_counter()
                self.fps_tracker['process'] = 10 / (current_time - last_time)
                last_time = current_time
                
            # GPU sync periodically - IMPORTANT TO KEEP
            if self.gpu_available and frame_count % 30 == 0:
                torch.cuda.synchronize(self.device)


    def _process_frame_gpu(self, frame_bgr: np.ndarray, frame_rgb: np.ndarray) -> Dict[str, Any]:
        """Process frame with GPU optimization."""
        results = {}
        
        # Emotion processing (already GPU optimized in handler)
        results['face_bbox'], results['emotion_info'] = self.emonet_handler.process_frame(frame_bgr)
        
        # Always process hands
        results['hand_info'] = self.hand_handler.process_frame(frame_rgb, results['face_bbox'])
        
        # Face-dependent processing
        if results['face_bbox'] is not None:
            # Iris detection
            results['iris_info'] = self.iris_handler.process_frame(frame_rgb)
            
            # Distance calculation (depends on iris)
            if results['iris_info']:
                results['distance_info'] = self.distance_handler.process_frame(
                    results['face_bbox'], results['iris_info'], frame_bgr.shape[:2]
                )
            else:
                results['distance_info'] = None
                
            # Pose estimation (process every frame at 10 FPS)
            results['pose_info'] = self.pose_handler.process_frame(frame_rgb)
        else:
            results['iris_info'] = None
            results['distance_info'] = None
            results['pose_info'] = None
            
        return results
        
    def start(self) -> None:
        """Start video stream and processing threads."""
        print("Starting GPU-optimized emotion recognition (10 FPS)...")
        print("Press 'q' to quit, 's' to save screenshot, 'd' to toggle debug")
        
        if self.gpu_available:
            gpu_name = torch.cuda.get_device_name(self.device)
            print(f"GPU: {gpu_name}")
            
        if self.feedback_handler:
            print("ChatGPT feedback enabled - Watch for supportive messages!")
        
        # Start session tracking
        self.report_handler.start_session()
        print("📊 Session metrics collection started")
            
        self.running = True
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
        # Main display loop
        self._display_loop()
        

    def _display_loop(self) -> None:
        """Main display loop running at target FPS."""
        frame_count = 0
        last_time = time.perf_counter()
        
        try:
            while self.running:
                loop_start = time.perf_counter()
                
                # Get frame for display (separate copy)
                frame = None
                with self.frame_lock:
                    if self.current_frame is not None:
                        frame = self.current_frame.copy()  # Dedicated copy for display
                        
                if frame is None:
                    time.sleep(0.01)
                    continue
                    
                # Work on copy, never modify original
                self.display_frame = frame  # No copy needed, frame is already a copy
                    
                # Get latest results
                with self.results_lock:
                    results = self.latest_results.copy()
                    
                # Verify result validity before drawing
                # If no face or hands detected, don't draw anything
                has_detection = (
                    results.get('face_bbox') is not None or 
                    (results.get('hand_info') and results['hand_info'].get('hands_detected', False))
                )
                
                if has_detection:
                    # Apply visualizations only if there's something detected
                    if results['iris_info'] is not None:
                        self.display_frame = self.iris_handler.draw_iris_visualization(
                            self.display_frame, results['iris_info'], self.logger.debug
                        )
                        
                    if results['pose_info'] is not None:
                        self.display_frame = self.pose_handler.draw_pose_visualization(
                            self.display_frame, results['pose_info'], self.logger.debug
                        )
                        
                    if results['hand_info'] is not None:
                        self.display_frame = self.hand_handler.draw_hand_visualization(
                            self.display_frame, results['hand_info'], self.logger.debug
                        )
                
                # Main visualization (always execute to show clean frame)
                self.display_frame = self.logger.create_visualization(
                    self.display_frame,
                    results['face_bbox'] if has_detection else None,
                    results['emotion_info'] if has_detection else None,
                    results['iris_info'] if has_detection else None,
                    results['distance_info'] if has_detection else None,
                    results['pose_info'] if has_detection else None,
                    results['hand_info'] if has_detection else None
                )
                
                # Feedback overlay
                if self.feedback_handler:
                    self.display_frame = self.feedback_handler.draw_feedback(self.display_frame)
                    
                # Performance overlay in debug mode
                if self.show_fps:
                    self._draw_performance_overlay(self.display_frame)
                    
                # Display frame
                cv2.imshow('Real-time Emotion Recognition', self.display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.logger.save_screenshot(self.display_frame)
                    print("Screenshot saved!")
                elif key == ord('d'):
                    self.logger.debug = not self.logger.debug
                    print(f"Debug mode: {'ON' if self.logger.debug else 'OFF'}")
                    
                # Update counters
                self.frame_counter += 1
                self.logger.increment_frame_counter()
                
                # FPS tracking
                frame_count += 1
                if frame_count % 10 == 0:
                    current_time = time.perf_counter()
                    self.fps_tracker['display'] = 10 / (current_time - last_time)
                    last_time = current_time
                    self.logger.update_fps()
                    
                # Debug output
                if self.logger.debug and frame_count % 30 == 0:
                    self.logger.print_debug_info(
                        results['face_bbox'],
                        results['emotion_info'],
                        results['iris_info'],
                        results['pose_info'],
                        results['distance_info'],
                        results['hand_info'],
                        self.device
                    )
                    
                # Frame timing to maintain target FPS
                elapsed = time.perf_counter() - loop_start
                sleep_time = self.frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.stop()

            
    def _draw_performance_overlay(self, frame: np.ndarray) -> None:
        """Draw performance statistics overlay."""
        if self.process_times:
            avg_process = np.mean(self.process_times) * 1000
            text = f"FPS - Cap: {self.fps_tracker['capture']:.1f} | Proc: {self.fps_tracker['process']:.1f} | Disp: {self.fps_tracker['display']:.1f} | Avg: {avg_process:.1f}ms"
            cv2.putText(frame, text, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                       
    def _print_performance_stats(self) -> None:
        """Print detailed performance statistics."""
        print("\n=== Performance Statistics ===")
        print(f"Capture FPS: {self.fps_tracker['capture']:.1f}")
        print(f"Process FPS: {self.fps_tracker['process']:.1f}")
        print(f"Display FPS: {self.fps_tracker['display']:.1f}")
        
        if self.process_times:
            times_ms = np.array(self.process_times) * 1000
            print(f"Processing time: avg={np.mean(times_ms):.1f}ms, max={np.max(times_ms):.1f}ms, min={np.min(times_ms):.1f}ms")
            
        if self.gpu_available:
            print(f"GPU Memory: {torch.cuda.memory_allocated(self.device) / 1e6:.1f} MB allocated")
            
    def stop(self) -> None:
        """Stop all threads and clean up resources."""
        print("\nStopping emotion recognition...")
        self.running = False
        
        # Wait for threads
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1)
            
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=1)
            
        # Stop feedback handler
        if self.feedback_handler:
            self.feedback_handler.stop()
            
        # Generate final report
        if len(self.report_handler.session_metrics) > 5:  # Only if sufficient data
            print("\n" + "="*50)
            print("GENERATING COACHING REPORT...")
            print("="*50)
            
            try:
                # Generate and save comprehensive report
                report_content = self.report_handler.generate_report(save_to_file=True)
                
                # Save raw session data for future analysis
                self.report_handler.save_session_data()
                
                # Show key highlights
                analysis = self.report_handler.analyze_session()
                if 'overall' in analysis:
                    score = analysis['overall'].get('overall_score', 0)
                    level = analysis['overall'].get('performance_level', 'unknown')
                    print(f"Overall Performance: {level.upper()} ({score:.2f}/1.0)")
                
                print("-" * 40)
                print("📁 Full report saved to file")
                print("✅ Session analysis complete!")
                
            except Exception as e:
                print(f"❌ Error generating final report: {e}")
                # Still save whatever data we have
                try:
                    self.report_handler.save_session_data()
                    print("📋 Raw session data saved for manual analysis")
                except:
                    pass
        else:
            print("⚠️  Insufficient data for comprehensive report (session too short)")
            
        # Release camera
        if self.cap:
            self.cap.release()
            
        # Clear GPU cache
        if self.gpu_available:
            torch.cuda.empty_cache()
            
        cv2.destroyAllWindows()
        
        # Print final stats
        self._print_performance_stats()
        print("Stopped successfully")
        
    def is_running(self) -> bool:
        """Check if video stream is running."""
        return self.running