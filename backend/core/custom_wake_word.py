"""
Custom Wake Word Detector - Real-time Detection
Uses trained PyTorch CNN model for "Hey Zenith" detection
Replaces Porcupine (no API key needed!)
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import Optional

# Import from training folder
import sys
sys.path.append(str(Path(__file__).parent.parent / "wake_word_training"))

from audio_features import AudioFeatureExtractor
from wake_word_model import create_wake_word_model

logger = logging.getLogger(__name__)


class CustomWakeWordDetector:
    """
    Real-time wake word detection using custom CNN model
    """
    
    def __init__(
        self,
        model_path: str = "models/custom_wake_word.pth",
        sample_rate: int = 16000,
        threshold: float = 0.7,  # Confidence threshold (70%)
        window_size: float = 1.5  # Audio window in seconds
    ):
        """
        Initialize custom wake word detector
        
        Args:
            model_path: Path to trained .pth model
            sample_rate: Audio sample rate (16kHz)
            threshold: Detection confidence threshold (0.0-1.0)
            window_size: Audio window size in seconds
        """
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.window_size = window_size
        self.window_samples = int(sample_rate * window_size)
        
        # Device selection (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Audio buffer for sliding window
        self.audio_buffer = np.array([], dtype=np.float32)
        
        # Feature extractor
        self.feature_extractor = AudioFeatureExtractor(sample_rate=sample_rate)
        
        # Load model
        self.model = None
        self._load_model()
        
        logger.info(f"✅ Custom wake word detector initialized")
        logger.info(f"   Model: {self.model_path.name}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Threshold: {threshold:.2f}")
    
    def _load_model(self):
        """Load trained model from disk"""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Wake word model not found: {self.model_path}\n"
                f"Please train the model first: python wake_word_training/2_train_model.py"
            )
        
        try:
            # Create model architecture
            self.model = create_wake_word_model(dropout_rate=0.3)
            
            # Load trained weights
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            # Set to evaluation mode
            self.model.eval()
            self.model.to(self.device)
            
            logger.info(f"✅ Wake word model loaded: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def process_audio(self, audio_chunk: np.ndarray) -> bool:
        """
        Process incoming audio and detect wake word
        
        Args:
            audio_chunk: Audio samples (numpy array)
        
        Returns:
            True if wake word detected, False otherwise
        """
        # Add new audio to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        
        # Keep only the sliding window (1.5 seconds)
        if len(self.audio_buffer) > self.window_samples:
            self.audio_buffer = self.audio_buffer[-self.window_samples:]
        
        # Need at least 1.5 seconds of audio
        if len(self.audio_buffer) < self.window_samples:
            return False
        
        # Extract features from current window
        try:
            features = self.feature_extractor.extract_features_from_audio(
                self.audio_buffer
            )
            
            # Add batch and channel dimensions: (39, 100) -> (1, 1, 39, 100)
            features = features[np.newaxis, np.newaxis, :]
            features_tensor = torch.FloatTensor(features).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.model(features_tensor)
                confidence = output.item()
            
            # Check if above threshold
            if confidence >= self.threshold:
                logger.info(f"🎯 Wake word detected! Confidence: {confidence:.2%}")
                # Clear buffer to avoid re-detection
                self.audio_buffer = np.array([], dtype=np.float32)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return False
    
    def set_threshold(self, threshold: float):
        """
        Adjust detection threshold
        
        Args:
            threshold: New threshold (0.0-1.0)
                     Lower = more sensitive (more false positives)
                     Higher = less sensitive (may miss wake words)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        self.threshold = threshold
        logger.info(f"Wake word threshold updated: {threshold:.2f}")
    
    def reset(self):
        """Reset audio buffer"""
        self.audio_buffer = np.array([], dtype=np.float32)


# ============================================
# TEST FUNCTION
# ============================================

if __name__ == "__main__":
    # Test the detector
    detector = CustomWakeWordDetector(
        model_path="../models/custom_wake_word.pth",
        threshold=0.7
    )
    
    print("✅ Custom wake word detector loaded successfully!")
    print(f"   Model size: {detector.model.count_parameters():,} parameters")
    print(f"   Device: {detector.device}")
    print("\nReady for real-time detection!")
