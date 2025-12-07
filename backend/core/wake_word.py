"""
Custom Wake Word Detection Module
Uses trained PyTorch CNN model for "Hey Zenith" detection
Version 5.1 - No Porcupine dependency
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """
    Wake word detection using custom PyTorch CNN model
    """
    
    def __init__(self, config: dict):
        """
        Initialize custom wake word detector
        
        Args:
            config: Wake word configuration from settings.yaml
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.keyword = config.get('keyword', 'hey zenith')
        
        self.detector = None
        self.is_listening = False
        
        if self.enabled:
            self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the custom wake word detector"""
        
        try:
            # Import custom wake word detector
            from core.custom_wake_word import CustomWakeWordDetector
            
            # Get configuration
            model_path = self.config.get('model_path', 'models/custom_wake_word.pth')
            threshold = self.config.get('threshold', 0.7)
            window_size = self.config.get('window_size', 1.5)
            
            # Convert to absolute path if relative
            if not Path(model_path).is_absolute():
                model_path = Path(__file__).parent.parent / model_path
            
            logger.info(f"🔧 Initializing custom wake word detector...")
            logger.info(f"   Model path: {model_path}")
            logger.info(f"   Threshold: {threshold}")
            logger.info(f"   Window size: {window_size}s")
            
            # Create detector
            self.detector = CustomWakeWordDetector(
                model_path=str(model_path),
                threshold=threshold,
                window_size=window_size
            )
            
            logger.info(f"✅ Custom wake word detector initialized")
            logger.info(f"   Keyword: '{self.keyword}'")
            logger.info(f"   Ready for detection!")
            
        except FileNotFoundError as e:
            logger.error(f"❌ Custom model not found: {e}")
            logger.error(f"   Expected location: {model_path}")
            logger.error(f"   Please train the model first:")
            logger.error(f"   cd backend")
            logger.error(f"   python wake_word_training/2_train_model.py")
            raise RuntimeError(
                f"Wake word model not found at: {model_path}\n"
                f"Please train the model first:\n"
                f"  cd backend\n"
                f"  python wake_word_training/2_train_model.py"
            )
        
        except ImportError as e:
            logger.error(f"❌ Custom wake word dependencies missing: {e}")
            logger.error(f"   Install required packages:")
            logger.error(f"   pip install torch librosa")
            raise RuntimeError(
                f"Missing dependencies for custom wake word: {e}\n"
                f"Install: pip install torch librosa"
            )
        
        except Exception as e:
            logger.error(f"❌ Custom wake word initialization failed: {e}")
            raise RuntimeError(f"Wake word initialization failed: {e}")
    
    def process(self, audio_data) -> bool:
        """
        Process audio data for wake word detection
        
        Args:
            audio_data: Audio samples (numpy array)
        
        Returns:
            True if wake word detected, False otherwise
        """
        if not self.enabled or self.detector is None:
            return False
        
        try:
            return self.detector.process_audio(audio_data)
        
        except Exception as e:
            logger.error(f"Error processing wake word: {e}")
            return False
    
    def start(self):
        """Start listening for wake word"""
        self.is_listening = True
        logger.info(f"👂 Listening for '{self.keyword}'")
    
    def stop(self):
        """Stop listening for wake word"""
        self.is_listening = False
        logger.info("🛑 Wake word detection stopped")
    
    def set_threshold(self, threshold: float):
        """
        Adjust detection threshold
        
        Args:
            threshold: New threshold value (0.0-1.0)
                     Lower = more sensitive (more false positives)
                     Higher = less sensitive (may miss wake words)
        """
        if self.detector and hasattr(self.detector, 'set_threshold'):
            self.detector.set_threshold(threshold)
            logger.info(f"🔧 Wake word threshold updated: {threshold}")
        else:
            logger.warning("⚠️  Threshold adjustment not available")
    
    def get_sample_rate(self) -> int:
        """
        Get required sample rate for wake word detection
        
        Returns:
            Sample rate in Hz (16000 for custom model)
        """
        return 16000
    
    def get_frame_length(self) -> int:
        """
        Get required frame length for wake word detection
        
        Returns:
            Frame length in samples
        """
        return 512
    
    def cleanup(self):
        """Clean up resources"""
        if self.detector:
            self.detector = None
        logger.info("Wake word detector cleanup complete")
