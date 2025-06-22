import cv2
import numpy as np
import time
from openai import OpenAI
from typing import Dict, Optional, List, Tuple
import threading
import queue
import json


class FeedbackHandler:
    """
    Handles ChatGPT-based feedback for emotion recognition system.
    Analyzes emotional changes and provides contextual feedback.
    """
    
    def __init__(self, 
                 api_key: str,
                 model: str = "gpt-4o-mini",
                 feedback_interval: float = 5.0,
                 max_feedback_history: int = 10):
        """
        Initialize feedback handler with ChatGPT API.
        
        Args:
            api_key: OpenAI API key
            model: ChatGPT model to use
            feedback_interval: Minimum seconds between feedback updates
            max_feedback_history: Maximum feedback messages to keep
        """
        self.api_key = api_key
        self.model = model
        self.feedback_interval = feedback_interval
        self.max_feedback_history = max_feedback_history
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Feedback management
        self.current_feedback = ""
        self.feedback_history = []
        self.last_feedback_time = 0
        self.feedback_queue = queue.Queue(maxsize=5)
        
        # State tracking for change detection
        self.previous_state = None
        self.negative_change_threshold = 0.2
        
        # Threading for async API calls
        self.processing_thread = None
        self.running = False
        
        # Visual settings
        self.feedback_display_duration = 10.0  # seconds
        self.feedback_start_time = 0
        
    def start(self):
        """Start the feedback processing thread."""
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_feedback_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def stop(self):
        """Stop the feedback processing thread."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
            
    def _process_feedback_loop(self):
        """Background thread for processing feedback requests."""
        while self.running:
            try:
                # Get state from queue
                state_data = self.feedback_queue.get(timeout=1)
                
                # Generate feedback using ChatGPT
                feedback = self._generate_feedback(state_data)
                
                if feedback:
                    self.current_feedback = feedback
                    self.feedback_start_time = time.time()
                    self.feedback_history.append({
                        'timestamp': time.time(),
                        'feedback': feedback,
                        'trigger': state_data.get('trigger', 'unknown')
                    })
                    
                    # Keep history limited
                    if len(self.feedback_history) > self.max_feedback_history:
                        self.feedback_history.pop(0)
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Feedback processing error: {e}")
                
    def analyze_state(self, 
                     emotion_info: Optional[Dict],
                     iris_info: Optional[Dict],
                     pose_info: Optional[Dict],
                     distance_info: Optional[Dict],
                     hand_info: Optional[Dict]) -> None:
        """
        Analyze current state for negative changes.
        
        Args:
            emotion_info: Current emotion data
            iris_info: Current iris tracking data
            pose_info: Current pose data
            distance_info: Current distance data
            hand_info: Current hand tracking data
        """
        current_time = time.time()
        
        # Check if enough time has passed since last feedback
        if current_time - self.last_feedback_time < self.feedback_interval:
            return
            
        # Build current state
        current_state = {
            'emotion': emotion_info,
            'iris': iris_info,
            'pose': pose_info,
            'distance': distance_info,
            'hand': hand_info,
            'timestamp': current_time
        }
        
        # Detect negative changes
        if self.previous_state is not None:
            negative_changes = self._detect_negative_changes(
                self.previous_state, current_state
            )
            
            if negative_changes:
                # Prepare data for ChatGPT
                state_data = {
                    'current_state': current_state,
                    'previous_state': self.previous_state,
                    'negative_changes': negative_changes,
                    'trigger': self._identify_trigger(negative_changes)
                }
                
                # Queue for processing (non-blocking)
                try:
                    self.feedback_queue.put_nowait(state_data)
                    self.last_feedback_time = current_time
                except queue.Full:
                    pass
                    
        self.previous_state = current_state
        
    def _detect_negative_changes(self, 
                                prev_state: Dict, 
                                curr_state: Dict) -> List[Dict]:
        """
        Detect negative changes between states.
        
        Returns:
            List of detected negative changes
        """
        changes = []
        
        # Check emotion changes
        if prev_state.get('emotion') and curr_state.get('emotion'):
            prev_valence = prev_state['emotion'].get('valence', 0)
            curr_valence = curr_state['emotion'].get('valence', 0)
            
            # Significant negative valence shift
            if curr_valence < prev_valence - self.negative_change_threshold:
                changes.append({
                    'type': 'emotion_negative',
                    'prev_valence': prev_valence,
                    'curr_valence': curr_valence,
                    'emotion': curr_state['emotion'].get('predicted_emotion', 'unknown')
                })
                
        # Check posture degradation
        if prev_state.get('pose') and curr_state.get('pose'):
            prev_score = prev_state['pose']['posture_metrics'].get('overall_posture_score', 1)
            curr_score = curr_state['pose']['posture_metrics'].get('overall_posture_score', 1)
            
            if curr_score < prev_score - 0.15:
                changes.append({
                    'type': 'posture_degradation',
                    'prev_score': prev_score,
                    'curr_score': curr_score
                })
                
        # Check distance issues
        if curr_state.get('distance'):
            status = curr_state['distance'].get('distance_status', '')
            if status in ['Too Close', 'Too Far']:
                changes.append({
                    'type': 'distance_issue',
                    'status': status,
                    'distance': curr_state['distance'].get('distance_cm', 0)
                })
                
        # Check face coverage
        if curr_state.get('hand'):
            interference = curr_state['hand'].get('face_interference', {})
            if interference.get('sustained_interference', False):
                changes.append({
                    'type': 'face_covered',
                    'duration': interference.get('duration', 0)
                })
                
        # Check eye strain indicators
        if curr_state.get('iris'):
            metrics = curr_state['iris'].get('eye_metrics', {})
            if metrics.get('eye_closure_percentage', 0) > 50:
                changes.append({
                    'type': 'eye_strain',
                    'closure_percentage': metrics['eye_closure_percentage'],
                    'blink_count': metrics.get('blink_count_last_5_sec', 0)
                })
                
        return changes
        
    def _identify_trigger(self, changes: List[Dict]) -> str:
        """Identify primary trigger from changes."""
        if not changes:
            return 'none'
            
        # Priority order for triggers
        for change in changes:
            if change['type'] == 'face_covered':
                return 'face_covered'
            elif change['type'] == 'emotion_negative':
                return 'negative_emotion'
            elif change['type'] == 'eye_strain':
                return 'eye_strain'
            elif change['type'] == 'posture_degradation':
                return 'poor_posture'
            elif change['type'] == 'distance_issue':
                return 'distance_problem'
                
        return changes[0]['type']
        
    def _generate_feedback(self, state_data: Dict) -> Optional[str]:
        """
        Generate feedback using ChatGPT API with shorter text requirement.
        """
        try:
            # Prepare context for ChatGPT
            context = self._prepare_context(state_data)
            print('context--------')
            print(context)

            # Create prompt with stricter length requirement
            prompt = f"""You are monitoring someone's emotional and physical state via webcam during communication.

            Based on the issue described below, give a **direct and corrective instruction** (max 12 words) to improve their non-verbal communication.

            Issue detected: {context}

            Instructions:
            - Be extremely concise (max 12 words total).
            - Use simple, clear commands.
            - Focus on ONE specific action to fix.
            - Avoid explanations or reasons.
            """

            # Create OpenAI client instance
            client = OpenAI(api_key=self.api_key)

            # Send request to the new API
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": (
                        "You are a strict communication coach. Give ultra-short commands (max 12 words). "
                        "Use imperative sentences. Be direct and clear. No explanations."
                    )},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=30,  # Reduced tokens for shorter responses
                temperature=0.5,
                timeout=5
            )

            feedback = response.choices[0].message.content.strip()
            
            # Additional safety check - truncate if still too long
            if len(feedback.split()) > 12:
                words = feedback.split()[:12]
                feedback = " ".join(words)
            
            return feedback

        except Exception as e:
            print(f"ChatGPT API error: {e}")
            return self._get_fallback_feedback(state_data.get('trigger', 'unknown'))
            
    def _prepare_context(self, state_data: Dict) -> str:
        """Prepare context string for ChatGPT."""
        changes = state_data.get('negative_changes', [])
        context_parts = []
        
        for change in changes:
            if change['type'] == 'emotion_negative':
                context_parts.append(
                    f"Emotion shifted to {change['emotion']} (valence dropped from "
                    f"{change['prev_valence']:.2f} to {change['curr_valence']:.2f})"
                )
            elif change['type'] == 'posture_degradation':
                context_parts.append(
                    f"Posture quality decreased from {change['prev_score']:.2f} to "
                    f"{change['curr_score']:.2f}"
                )
            elif change['type'] == 'distance_issue':
                context_parts.append(
                    f"Person is {change['status'].lower()} at {change['distance']:.0f}cm"
                )
            elif change['type'] == 'face_covered':
                context_parts.append(
                    f"Face has been covered for {change['duration']:.1f} seconds"
                )
            elif change['type'] == 'eye_strain':
                context_parts.append(
                    f"Eyes showing strain ({change['closure_percentage']:.0f}% closed, "
                    f"{change['blink_count']} blinks in 5 sec)"
                )
                
        return '\n'.join(context_parts)
        
    def _get_fallback_feedback(self, trigger: str) -> str:
        """Get shorter fallback feedback when API fails."""
        fallback_messages = {
            'negative_emotion': "Breathe deeply, stay positive",
            'poor_posture': "Sit up straight",
            'distance_problem': "Adjust your distance",
            'face_covered': "Uncover your face",
            'eye_strain': "Rest your eyes",
            'unknown': "Stay focused"
        }
        return fallback_messages.get(trigger, fallback_messages['unknown'])
        
    def _wrap_text(self, text: str, font, scale: float, thickness: int, max_width: int) -> List[str]:
        """
        Wrap text to fit within specified width.
        
        Args:
            text: Text to wrap
            font: OpenCV font
            scale: Font scale
            thickness: Font thickness
            max_width: Maximum width in pixels
            
        Returns:
            List of text lines
        """
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            # Try adding word to current line
            test_line = current_line + (" " if current_line else "") + word
            (test_w, _), _ = cv2.getTextSize(test_line, font, scale, thickness)
            
            if test_w <= max_width:
                current_line = test_line
            else:
                # Word doesn't fit, start new line
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    # Single word is too long, split it
                    lines.append(word[:len(word)//2] + "-")
                    current_line = word[len(word)//2:]
        
        if current_line:
            lines.append(current_line)
        
        return lines
        
    def draw_feedback(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw feedback on frame with adaptive text sizing and wrapping.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with feedback overlay
        """
        # Check if feedback should still be displayed
        current_time = time.time()
        if (not self.current_feedback or 
            current_time - self.feedback_start_time > self.feedback_display_duration):
            return frame
            
        # Prepare feedback text
        feedback_text = self.current_feedback
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Frame dimensions
        frame_height, frame_width = frame.shape[:2]
        max_text_width = frame_width - 40  # Leave 20px margin on each side
        
        # Try different font sizes to find the best fit
        for scale in [0.8, 0.7, 0.6, 0.5, 0.4]:
            thickness = 2 if scale >= 0.6 else 1
            
            # Check if text fits in one line
            (text_w, text_h), _ = cv2.getTextSize(feedback_text, font, scale, thickness)
            
            if text_w <= max_text_width:
                # Text fits in one line
                lines = [feedback_text]
                total_text_height = text_h
                break
            else:
                # Try to wrap text
                lines = self._wrap_text(feedback_text, font, scale, thickness, max_text_width)
                total_text_height = len(lines) * (text_h + 5)  # 5px line spacing
                
                # Check if wrapped text fits in reasonable height
                if total_text_height <= frame_height * 0.3:  # Max 30% of frame height
                    break
        
        # Calculate box dimensions
        padding = 15
        box_height = total_text_height + padding * 2
        box_y = max(10, frame_height - box_height - 20)  # Ensure it doesn't go off screen
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (10, box_y), 
                     (frame_width - 10, box_y + box_height), 
                     (255, 255, 255), 
                     -1)
        
        # Add transparency
        alpha = 0.9
        frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        # Draw border
        cv2.rectangle(frame, 
                     (10, box_y), 
                     (frame_width - 10, box_y + box_height), 
                     (200, 200, 200), 
                     2)
        
        # Draw text lines
        for i, line in enumerate(lines):
            (line_w, line_h), _ = cv2.getTextSize(line, font, scale, thickness)
            text_x = (frame_width - line_w) // 2  # Center each line
            text_y = box_y + padding + line_h + i * (line_h + 5)
            
            cv2.putText(frame, line, 
                       (text_x, text_y), 
                       font, scale, 
                       (0, 0, 0),  # Black text
                       thickness)
        
        # Add fade effect for last 2 seconds
        elapsed = current_time - self.feedback_start_time
        if elapsed > self.feedback_display_duration - 2:
            fade_alpha = (self.feedback_display_duration - elapsed) / 2
            fade_overlay = frame.copy()
            cv2.rectangle(fade_overlay, 
                         (10, box_y), 
                         (frame_width - 10, box_y + box_height), 
                         (255, 255, 255), 
                         -1)
            frame = cv2.addWeighted(frame, 1, fade_overlay, fade_alpha * 0.3, 0)
            
        return frame
        
    def get_feedback_history(self) -> List[Dict]:
        """Get feedback history."""
        return self.feedback_history.copy()