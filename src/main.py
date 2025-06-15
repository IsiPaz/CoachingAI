#!/usr/bin/env python3
"""
Real-time emotion recognition system using EmoNet.
Refactored into modular components for better maintainability.
"""

import argparse
from emonet_handler import EmoNetHandler
from emotion_logger import EmotionLogger
from video_stream import VideoStream


class RealtimeEmotionRecognizer:
    """
    Main coordinator class for real-time emotion recognition system.
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
            show_fps: Whether to display FPS counter
            show_circumplex: Whether to display circumplex visualization
            debug: Whether to enable debug mode
        """
        print("Initializing Real-time Emotion Recognition System...")
        
        # Initialize components
        self.emonet_handler = EmoNetHandler(
            n_expression=n_expression,
            device=device,
            image_size=image_size
        )
        
        self.logger = EmotionLogger(
            debug=debug,
            show_fps=show_fps,
            show_circumplex=show_circumplex
        )
        
        self.video_stream = VideoStream(
            emonet_handler=self.emonet_handler,
            logger=self.logger,
            camera_id=camera_id,
            target_fps=target_fps
        )
        
        print("System initialized successfully!")
        
    def start(self) -> None:
        """Start the real-time emotion recognition system."""
        self.video_stream.start()
        
    def stop(self) -> None:
        """Stop the real-time emotion recognition system."""
        self.video_stream.stop()
        
    def is_running(self) -> bool:
        """Check if the system is running."""
        return self.video_stream.is_running()


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Real-time Emotion Recognition")
    
    parser.add_argument("--n_expression", type=int, default=8, choices=[5, 8],
                       help="Number of emotion classes (5 or 8)")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to run models on (cuda:0, cpu)")
    parser.add_argument("--image_size", type=int, default=256,
                       help="Size for emotion recognition input")
    parser.add_argument("--camera_id", type=int, default=0,
                       help="Camera device ID")
    parser.add_argument("--target_fps", type=int, default=30,
                       help="Target FPS for processing")
    parser.add_argument("--show_fps", action="store_true",
                       help="Show FPS counter")
    parser.add_argument("--show_circumplex", action="store_true",
                       help="Show circumplex visualization")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with detailed console output")
    
    args = parser.parse_args()
    
    try:
        # Initialize and start the system
        recognizer = RealtimeEmotionRecognizer(
            n_expression=args.n_expression,
            device=args.device,
            image_size=args.image_size,
            camera_id=args.camera_id,
            target_fps=args.target_fps,
            show_fps=args.show_fps,
            show_circumplex=args.show_circumplex,
            debug=args.debug
        )
        
        recognizer.start()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the EmoNet model files are in the correct location.")
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Please check your camera connection and device settings.")
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()