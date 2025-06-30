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
                 feedback_interval: float = 15.0,
                 max_feedback_history: int = 20):
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

        self.last_state_signature = None
        self.last_feedback = ""

        # Feedback management
        self.current_feedback = ""
        self.feedback_history = []
        self.last_feedback_time = 0
        self.feedback_queue = queue.Queue(maxsize=5)
        
        # Context system
        self.context_window = 15.0  # 15 seconds of context
        self.state_history = []  # State History
        self.feedback_diversity_tracker = []  # To avoid repetitions

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


    def _get_state_signature(self, state: Dict) -> str:
        """
        Creates a lightweight hash of key state metrics to detect state changes.
        """
        key_metrics = [
            state['interpretations'].get('emotional_state', {}).get('primary_emotion', ''),
            state['interpretations'].get('attention_quality', {}).get('overall_engagement', ''),
            state['interpretations'].get('physical_presence', {}).get('posture_quality', ''),
            state['interpretations'].get('gestural_communication', {}).get('gesture_appropriateness', False)
        ]
        return "|".join(map(str, key_metrics))
                    

    def _process_feedback_loop(self):
        """Background thread for feedback processing"""
        while self.running:
            try:
                context_data = self.feedback_queue.get(timeout=1)

                # Limit history for memory
                if len(context_data['state_history']) > 20:
                    context_data['state_history'] = context_data['state_history'][-15:]

                state_signature = self._get_state_signature(context_data['current_state'])

                if state_signature == self.last_state_signature:
                    # Same state, keep previous feedback but update time to display
                    if self.current_feedback == "":
                        # If no previous feedback, force fallback
                        self.current_feedback = self._get_contextual_fallback(context_data)
                    self.feedback_start_time = time.time()
                    continue

                feedback = self._generate_feedback(context_data)

                if feedback:
                    self.current_feedback = feedback
                    self.feedback_start_time = time.time()

                    self.feedback_history.append({
                        'timestamp': time.time(),
                        'feedback': feedback,
                        'context_summary': self._get_brief_context_summary(context_data)
                    })

                    # Keep limited history
                    if len(self.feedback_history) > self.max_feedback_history:
                        self.feedback_history.pop(0)

                    self.last_state_signature = state_signature
                    self.last_feedback = feedback
                else:
                    # No feedback generated, clear current feedback to show nothing
                    self.current_feedback = ""
                    self.feedback_start_time = 0
                    self.last_feedback = ""

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Feedback processing error: {e}")


    def _get_brief_context_summary(self, context_data: Dict) -> str:
        """Brief context summary for history."""
        current = context_data['current_state']
        summary_parts = []
        
        if 'emotional_state' in current.get('interpretations', {}):
            emotion = current['interpretations']['emotional_state']['primary_emotion']
            summary_parts.append(f"emotion:{emotion}")
        
        if 'attention_quality' in current.get('interpretations', {}):
            engagement = current['interpretations']['attention_quality']['overall_engagement']
            summary_parts.append(f"attention:{engagement}")
        
        return ", ".join(summary_parts)
                
    def analyze_state(self, 
                    emotion_info: Optional[Dict],
                    iris_info: Optional[Dict],
                    pose_info: Optional[Dict],
                    distance_info: Optional[Dict],
                    hand_info: Optional[Dict]) -> None:
        """
        Analyzes the complete state in a contextual manner.
        """
        current_time = time.time()
        
        # Check feedback interval
        if current_time - self.last_feedback_time < self.feedback_interval:
            return
        
        # Build full contextual state
        current_state = self._build_contextual_state(
            emotion_info, iris_info, pose_info, distance_info, hand_info, current_time
        )
        
        # Add to history
        self.state_history.append(current_state)
        
        # Keep context window (last 15 seconds)
        cutoff_time = current_time - self.context_window
        self.state_history = [s for s in self.state_history if s['timestamp'] > cutoff_time]
        
        # Only generate feedback if there is enough context
        if len(self.state_history) >= 3:  # At least 3 states for context
            context_data = {
                'current_state': current_state,
                'state_history': self.state_history,
                'previous_feedback': self.feedback_history[-5:] if self.feedback_history else []
            }
            
            # Queue for processing
            try:
                self.feedback_queue.put_nowait(context_data)
                self.last_feedback_time = current_time
            except queue.Full:
                pass

    def _build_contextual_state(self, emotion_info, iris_info, pose_info, distance_info, hand_info, timestamp):
        """
        Builds a rich contextual state with qualitative interpretations.
        """
        state = {
            'timestamp': timestamp,
            'raw_data': {
                'emotion': emotion_info,
                'iris': iris_info,
                'pose': pose_info,
                'distance': distance_info,
                'hand': hand_info
            }
        }
        
        # Qualitative interpretations
        state['interpretations'] = {}
        
        # Contextual emotional analysis
        if emotion_info:
            valence = emotion_info.get('valence', 0)
            arousal = emotion_info.get('arousal', 0)
            emotion = emotion_info.get('predicted_emotion', 'neutral')
            
            state['interpretations']['emotional_state'] = {
                'primary_emotion': emotion,
                'energy_level': 'high' if arousal > 0.3 else 'low' if arousal < -0.3 else 'moderate',
                'mood_quality': 'positive' if valence > 0.2 else 'negative' if valence < -0.2 else 'neutral',
                'emotional_intensity': abs(valence) + abs(arousal)
            }
        
        # Presence and attention analysis
        if iris_info and distance_info:
            gaze_centered = iris_info['iris_position']['average_centering'] < 0.3
            eye_openness = iris_info['average_eye_aperture']
            distance_status = distance_info.get('distance_status', 'Unknown')
            
            state['interpretations']['attention_quality'] = {
                'gaze_focused': gaze_centered,
                'alertness_level': 'high' if eye_openness > 0.25 else 'low' if eye_openness < 0.15 else 'moderate',
                'distance_appropriateness': distance_status,
                'overall_engagement': 'engaged' if gaze_centered and eye_openness > 0.2 else 'disengaged'
            }
        
        # Posture and physical presence analysis
        if pose_info:
            posture_score = pose_info['posture_metrics'].get('overall_posture_score', 0.5)
            orientation = pose_info['posture_metrics'].get('orientation', 'Unknown')
            
            state['interpretations']['physical_presence'] = {
                'posture_quality': 'excellent' if posture_score > 0.7 else 'good' if posture_score > 0.5 else 'poor',
                'body_orientation': orientation.lower(),
                'physical_confidence': posture_score
            }
        
        # Gestural communication analysis
        if hand_info:
            hands_visible = hand_info.get('hands_detected', False)
            face_covered = hand_info.get('face_interference', {}).get('sustained_interference', False)
            
            state['interpretations']['gestural_communication'] = {
                'hands_active': hands_visible,
                'face_accessibility': not face_covered,
                'gesture_appropriateness': hands_visible and not face_covered
            }
        
        return state   

    def _generate_feedback(self, context_data: Dict) -> Optional[str]:
        current = context_data['current_state']
        interpretations = current.get('interpretations', {})

        positives = {
            'emotion': interpretations.get('emotional_state', {}).get('mood_quality') == 'positive',
            'posture': interpretations.get('physical_presence', {}).get('posture_quality') in ['excellent', 'good'],
            'gaze': interpretations.get('attention_quality', {}).get('gaze_focused') is True,
            'gesture_ok': interpretations.get('gestural_communication', {}).get('gesture_appropriateness') is True,
            'face_visible': interpretations.get('gestural_communication', {}).get('face_accessibility') is True,
            'distance_ok': interpretations.get('attention_quality', {}).get('distance_appropriateness') in ['ideal', 'ok']
        }

        negatives = {
            'emotion': interpretations.get('emotional_state', {}).get('mood_quality') == 'negative',
            'posture': interpretations.get('physical_presence', {}).get('posture_quality') == 'poor',
            'gaze': interpretations.get('attention_quality', {}).get('gaze_focused') is False,
            'gesture_bad': interpretations.get('gestural_communication', {}).get('gesture_appropriateness') is False,
            'face_covered': interpretations.get('gestural_communication', {}).get('face_accessibility') is False,
            'distance_bad': interpretations.get('attention_quality', {}).get('distance_appropriateness') == 'too_close'
        }

        positive_signals = sum(1 for v in positives.values() if v)
        negative_signals = sum(1 for v in negatives.values() if v)

        if positive_signals < 2 and negative_signals == 0:
            return None

        try:
            context_summary = self._prepare_contextual_prompt(context_data)
            recent_feedback_words = self._extract_recent_keywords()

            if negative_signals >= 1:
                tone = ("You are a strict but constructive communication coach. "
                        "Give clear, actionable feedback on what to improve, "
                        "while being encouraging.")
            else:
                tone = ("You are a positive and motivating communication coach. "
                        "Praise what is done well and encourage to keep it up.")

            prompt = f"""You observe a person in a video call.

    CURRENT CONTEXT:
    {context_summary}

    Instructions:
    - Analyze everything carefully.
    - Give clear, specific feedback in one or two short sentences.
    - Explicitly state if the person is doing well or needs improvement.
    - Avoid repeating these recent phrases: {', '.join(recent_feedback_words)}
    - Use natural, encouraging or corrective English.
    - Avoid jargon or technical terms.
    - Be concise but clear.
    - Use no more than 14 words, strictly. Do not exceed this limit.
    - Do not wrap your answer in quotation marks.

    Examples:
    - "Your posture and eye contact are great, keep it up!"
    - "Please straighten your back and show your face more."
    - "Good job staying engaged, just adjust your distance slightly."
    - "Your gestures distract a bit; try to keep hands visible but relaxed."
    """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": tone},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=60,
                temperature=0.7,
                timeout=5
            )

            feedback = response.choices[0].message.content.strip()

            # Post-processing to remove external quotes
            if feedback.startswith('"') and feedback.endswith('"'):
                feedback = feedback[1:-1].strip()

            self._update_diversity_tracker(feedback)
            return feedback

        except Exception as e:
            print(f"ChatGPT API error: {e}")
            return self._get_contextual_fallback(context_data)


    def _prepare_contextual_prompt(self, context_data: Dict) -> str:
        """
        Prepares a rich contextual summary for ChatGPT.
        """
        current = context_data['current_state']
        history = context_data['state_history']
        
        # Extract trends
        emotional_trend = self._analyze_emotional_trend(history)
        attention_trend = self._analyze_attention_trend(history)
        posture_trend = self._analyze_posture_trend(history)
        
        # Build contextual narrative
        context_parts = []
        
        # Current emotional state and trend
        if 'emotional_state' in current['interpretations']:
            emotion_data = current['interpretations']['emotional_state']
            context_parts.append(
                f"EMOTIONAL STATE: {emotion_data['primary_emotion']} with {emotion_data['energy_level']} energy "
                f"and {emotion_data['mood_quality']} mood. Trend: {emotional_trend}"
            )
        
        # Attention quality
        if 'attention_quality' in current['interpretations']:
            attention_data = current['interpretations']['attention_quality']
            context_parts.append(
                f"ATTENTION: {'Focused' if attention_data['gaze_focused'] else 'Distracted'}, "
                f"{attention_data['alertness_level']} alertness, {attention_data['distance_appropriateness']} distance. "
                f"Trend: {attention_trend}"
            )
        
        # Physical presence
        if 'physical_presence' in current['interpretations']:
            physical_data = current['interpretations']['physical_presence']
            context_parts.append(
                f"POSTURE: {physical_data['posture_quality']}, {physical_data['body_orientation']} orientation. "
                f"Trend: {posture_trend}"
            )
        
        # Gestural communication
        if 'gestural_communication' in current['interpretations']:
            gesture_data = current['interpretations']['gestural_communication']
            gesture_status = "appropriate" if gesture_data['gesture_appropriateness'] else "problematic"
            face_status = "visible" if gesture_data['face_accessibility'] else "covered"
            context_parts.append(f"GESTURES: {gesture_status}, face {face_status}")
        
        return '\n'.join(context_parts)

    def _analyze_emotional_trend(self, history: List[Dict]) -> str:
        """Analyzes emotional trend."""
        if len(history) < 2:
            return "stable"
        
        recent_emotions = []
        for state in history[-3:]:
            if 'emotional_state' in state.get('interpretations', {}):
                intensity = state['interpretations']['emotional_state'].get('emotional_intensity', 0)
                recent_emotions.append(intensity)
        
        if len(recent_emotions) >= 2:
            if recent_emotions[-1] > recent_emotions[0] + 0.3:
                return "improving"
            elif recent_emotions[-1] < recent_emotions[0] - 0.3:
                return "deteriorating"
        
        return "stable"

    def _analyze_attention_trend(self, history: List[Dict]) -> str:
        """Analyzes attention trend."""
        if len(history) < 2:
            return "stable"
        
        engagement_scores = []
        for state in history[-3:]:
            if 'attention_quality' in state.get('interpretations', {}):
                engaged = state['interpretations']['attention_quality'].get('overall_engagement') == 'engaged'
                engagement_scores.append(1.0 if engaged else 0.0)
        
        if len(engagement_scores) >= 2:
            avg_recent = np.mean(engagement_scores[-2:])
            avg_earlier = np.mean(engagement_scores[:-1]) if len(engagement_scores) > 2 else engagement_scores[0]
            
            if avg_recent > avg_earlier + 0.3:
                return "improving"
            elif avg_recent < avg_earlier - 0.3:
                return "deteriorating"
        
        return "stable"

    def _analyze_posture_trend(self, history: List[Dict]) -> str:
        """Analyzes posture trend."""
        if len(history) < 2:
            return "stable"
        
        posture_scores = []
        for state in history[-3:]:
            if 'physical_presence' in state.get('interpretations', {}):
                confidence = state['interpretations']['physical_presence'].get('physical_confidence', 0.5)
                posture_scores.append(confidence)
        
        if len(posture_scores) >= 2:
            if posture_scores[-1] > posture_scores[0] + 0.15:
                return "improving"
            elif posture_scores[-1] < posture_scores[0] - 0.15:
                return "deteriorating"
        
        return "stable"

    def _extract_recent_keywords(self) -> List[str]:
        """Extracts keywords from recent feedback for diversity."""
        keywords = []
        for feedback_entry in self.feedback_history[-3:]:  # Last 3 feedbacks
            feedback_text = feedback_entry.get('feedback', '')
            # Extract important words (not articles/prepositions)
            words = feedback_text.lower().split()
            important_words = [w for w in words if len(w) > 3 and w not in ['with', 'your', 'that', 'this', 'from']]
            keywords.extend(important_words)
        
        return keywords

    def _update_diversity_tracker(self, feedback: str) -> None:
        """Updates the diversity tracker."""
        self.feedback_diversity_tracker.append({
            'feedback': feedback,
            'timestamp': time.time(),
            'words': feedback.lower().split()
        })
        
        # Keep only the last 10 feedbacks
        if len(self.feedback_diversity_tracker) > 10:
            self.feedback_diversity_tracker.pop(0)

    def _get_contextual_fallback(self, context_data: Dict) -> str:
        """Smarter contextual fallback."""
        current = context_data['current_state']
        
        # Priorities for fallback
        if 'gestural_communication' in current['interpretations']:
            if not current['interpretations']['gestural_communication']['face_accessibility']:
                return "Uncover your face for better communication"
        
        if 'attention_quality' in current['interpretations']:
            if current['interpretations']['attention_quality']['overall_engagement'] == 'disengaged':
                return "Improve your eye contact and attention"
        
        if 'physical_presence' in current['interpretations']:
            quality = current['interpretations']['physical_presence']['posture_quality']
            if quality == 'poor':
                return "Straighten your posture and stay upright"
        
        return "Maintain focus and professional presence"

    def _wrap_text(self, text: str, font, scale: float, thickness: int, max_width: int) -> List[str]:
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            (test_w, _), _ = cv2.getTextSize(test_line, font, scale, thickness)

            if test_w <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

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