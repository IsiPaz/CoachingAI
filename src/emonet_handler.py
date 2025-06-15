import cv2
import numpy as np
import torch
from torch import nn
from pathlib import Path
from typing import Dict, Optional, List

from face_alignment.detection.sfd.sfd_detector import SFDDetector
from emonet.models import EmoNet


class EmoNetHandler:
    """
    Handles EmoNet model loading, face detection, and emotion recognition.
    """
    
    def __init__(self, 
                 n_expression: int = 8,
                 device: str = "cuda:0",
                 image_size: int = 256):
        """
        Initialize the EmoNet handler.
        
        Args:
            n_expression: Number of emotion classes (5 or 8)
            device: Device to run models on (cuda:0, cpu)
            image_size: Size for emotion recognition input
        """
        self.device = device
        self.image_size = image_size
        
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
        
        # Initialize models
        self._load_models(n_expression)
        
    def _load_models(self, n_expression: int) -> None:
        """Load EmoNet and face detection models with GPU optimization."""
        print(f"Loading models on device: {self.device}")
        
        # Load EmoNet
        state_dict_path = Path(__file__).parent.joinpath("emonet",
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
        
    def detect_faces(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
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
        
    def recognize_emotion(self, face_crop_rgb: np.ndarray) -> Dict[str, torch.Tensor]:
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
        
    def get_emotion_info(self, emotion_result: Dict[str, torch.Tensor]) -> Dict:
        """
        Extract emotion information from model output.
        
        Args:
            emotion_result: Raw emotion model output
            
        Returns:
            Dictionary with processed emotion information
        """
        # Get emotion probabilities
        emotion_probs = nn.functional.softmax(emotion_result["expression"], dim=1)
        predicted_class = torch.argmax(emotion_probs).cpu().item()
        confidence = emotion_probs[0, predicted_class].cpu().item()
        
        # Get valence and arousal
        valence = emotion_result["valence"].clamp(-1.0, 1.0).cpu().item()
        arousal = emotion_result["arousal"].clamp(-1.0, 1.0).cpu().item()
        
        # Get all emotion probabilities
        all_probs = {
            self.emotion_classes[i]: prob.cpu().item() 
            for i, prob in enumerate(emotion_probs[0])
        }
        
        return {
            "predicted_emotion": self.emotion_classes[predicted_class],
            "confidence": confidence,
            "valence": valence,
            "arousal": arousal,
            "all_probabilities": all_probs,
            "predicted_class": predicted_class
        }
        
    def process_frame(self, frame_bgr: np.ndarray) -> tuple:
        """
        Process a single frame for emotion recognition.
        
        Args:
            frame_bgr: Input frame in BGR format
            
        Returns:
            Tuple of (face_bbox, emotion_info) or (None, None) if no face detected
        """
        # Convert to RGB for face detection
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        detected_faces = self.detect_faces(frame_bgr)
        
        if len(detected_faces) > 0:
            # Use first detected face
            face_bbox = np.array(detected_faces[0]).astype(np.int32)
            
            # Extract face crop
            face_crop_rgb = frame_rgb[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
            
            if face_crop_rgb.size > 0:
                # Run emotion recognition
                emotion_result = self.recognize_emotion(face_crop_rgb)
                emotion_info = self.get_emotion_info(emotion_result)
                return face_bbox, emotion_info
                
        return None, None