"""
Speech Recognition Module - Whisper Integration
✅ v4.5: GPU Support + Dynamic Switching + Small Model
Supports: auto GPU detection, runtime GPU toggle, memory management
"""

import logging
import numpy as np
import torch
from faster_whisper import WhisperModel
from typing import Optional

logger = logging.getLogger(__name__)


class SpeechRecognizer:
    """
    Speech-to-text using Faster Whisper
    ✅ NEW: GPU support with dynamic switching
    """
    
    def __init__(self, config: dict):
        """
        Initialize Whisper model with GPU support
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.speech_config = config.get('speech', {})
        
        # Model settings
        self.model_name = self.speech_config.get('model', 'small')  # ✅ Default to small
        self.language = self.speech_config.get('language', 'en')
        
        # ✅ NEW: GPU Settings
        self.device_preference = self.speech_config.get('device', 'auto')
        self.compute_type = self.speech_config.get('compute_type', 'float16')
        
        gpu_settings = self.speech_config.get('gpu_settings', {})
        self.enable_gpu = gpu_settings.get('enable_gpu', True)
        self.fallback_to_cpu = gpu_settings.get('fallback_to_cpu', True)
        self.dynamic_switching = gpu_settings.get('dynamic_switching', True)
        self.max_gpu_memory = gpu_settings.get('max_gpu_memory', '3GB')
        
        # Transcription settings
        self.beam_size = self.speech_config.get('beam_size', 5)
        self.vad_filter = self.speech_config.get('vad_filter', True)
        self.best_of = self.speech_config.get('best_of', 5)
        
        # Initialize model
        self.model = None
        self.current_device = None
        self.is_gpu_available = self._check_gpu_availability()
        
        self._load_model()
    
    def _check_gpu_availability(self) -> bool:
        """
        Check if CUDA GPU is available
        
        Returns:
            True if GPU available and enabled
        """
        try:
            if not self.enable_gpu:
                logger.info("GPU disabled in config")
                return False
            
            if not torch.cuda.is_available():
                logger.info("CUDA not available")
                return False
            
            # Check VRAM
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            
            logger.info(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"   VRAM: {gpu_memory_gb:.1f} GB")
            
            if gpu_memory_gb < 2:
                logger.warning("⚠️  GPU has less than 2GB VRAM, using CPU")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"GPU check failed: {e}")
            return False
    
    def _determine_device(self) -> tuple:
        """
        Determine which device and compute type to use
        
        Returns:
            Tuple of (device_name, compute_type)
        """
        # User explicitly requested device
        if self.device_preference in ['cpu', 'cuda']:
            device = self.device_preference
            compute = 'int8' if device == 'cpu' else 'float16'
            return (device, compute)
        
        # Auto-detect
        if self.is_gpu_available:
            logger.info("🎮 Using GPU acceleration (CUDA)")
            return ('cuda', 'float16')
        else:
            logger.info("🖥️  Using CPU mode")
            return ('cpu', 'int8')
    
    def _load_model(self):
        """Load Whisper model with appropriate device"""
        try:
            device, compute_type = self._determine_device()
            
            print(f"🎧 Loading Whisper model ({self.model_name})...")
            
            # ✅ Load model with device settings
            self.model = WhisperModel(
                self.model_name,
                device=device,
                compute_type=compute_type,
                num_workers=4,
                cpu_threads=4
            )
            
            self.current_device = device
            
            # Print success message
            device_emoji = "🎮" if device == "cuda" else "🖥️"
            print(f"✅ Whisper {self.model_name} loaded on {device_emoji} {device.upper()}")
            
            if device == "cuda":
                print(f"   Compute: {compute_type}, VRAM Usage: ~1-2GB")
            else:
                print(f"   Compute: {compute_type}, RAM Usage: ~500MB")
            
            logger.info(f"Whisper model loaded: {self.model_name} on {device}")
            
        except Exception as e:
            logger.error(f"Whisper load error: {e}")
            
            # ✅ Fallback to CPU if GPU fails
            if self.fallback_to_cpu and device == 'cuda':
                logger.info("Falling back to CPU...")
                print("⚠️  GPU load failed, falling back to CPU...")
                
                try:
                    self.model = WhisperModel(
                        self.model_name,
                        device='cpu',
                        compute_type='int8',
                        num_workers=4,
                        cpu_threads=4
                    )
                    self.current_device = 'cpu'
                    print(f"✅ Whisper {self.model_name} loaded on CPU")
                    
                except Exception as e2:
                    logger.error(f"CPU fallback also failed: {e2}")
                    raise RuntimeError(f"Failed to load Whisper: {e2}")
            else:
                raise
    
    def transcribe(self, audio_data: np.ndarray) -> Optional[str]:
        """
        Transcribe audio to text
        
        Args:
            audio_data: Audio samples (16kHz, mono, int16)
        
        Returns:
            Transcribed text or None
        """
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return None
            
            # Convert int16 to float32
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Transcribe
            segments, info = self.model.transcribe(
                audio_float,
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                best_of=self.best_of,
                temperature=0.0
            )
            
            # Combine segments
            text = " ".join([segment.text for segment in segments]).strip()
            
            if text:
                logger.info(f"✅ Transcribed: {text}")
                return text
            else:
                logger.debug("No speech detected")
                return None
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
    
    # ✅ NEW: Dynamic GPU Switching Methods
    
    def switch_to_gpu(self) -> bool:
        """
        Switch to GPU mode at runtime
        
        Returns:
            True if successful
        """
        if not self.dynamic_switching:
            logger.warning("Dynamic switching disabled in config")
            return False
        
        if not self.is_gpu_available:
            logger.warning("GPU not available")
            return False
        
        if self.current_device == 'cuda':
            logger.info("Already using GPU")
            return True
        
        try:
            logger.info("Switching to GPU...")
            print("🎮 Switching Whisper to GPU...")
            
            self.model = WhisperModel(
                self.model_name,
                device='cuda',
                compute_type='float16',
                num_workers=4,
                cpu_threads=4
            )
            
            self.current_device = 'cuda'
            print("✅ Switched to GPU mode")
            logger.info("Switched to GPU successfully")
            return True
            
        except Exception as e:
            logger.error(f"GPU switch failed: {e}")
            print(f"❌ GPU switch failed: {e}")
            return False
    
    def switch_to_cpu(self) -> bool:
        """
        Switch to CPU mode at runtime
        
        Returns:
            True if successful
        """
        if not self.dynamic_switching:
            logger.warning("Dynamic switching disabled in config")
            return False
        
        if self.current_device == 'cpu':
            logger.info("Already using CPU")
            return True
        
        try:
            logger.info("Switching to CPU...")
            print("🖥️  Switching Whisper to CPU...")
            
            self.model = WhisperModel(
                self.model_name,
                device='cpu',
                compute_type='int8',
                num_workers=4,
                cpu_threads=4
            )
            
            self.current_device = 'cpu'
            print("✅ Switched to CPU mode")
            logger.info("Switched to CPU successfully")
            return True
            
        except Exception as e:
            logger.error(f"CPU switch failed: {e}")
            print(f"❌ CPU switch failed: {e}")
            return False
    
    def get_device_info(self) -> dict:
        """
        Get current device information
        
        Returns:
            Dictionary with device details
        """
        info = {
            'current_device': self.current_device,
            'model': self.model_name,
            'gpu_available': self.is_gpu_available,
            'dynamic_switching_enabled': self.dynamic_switching
        }
        
        if self.current_device == 'cuda':
            try:
                info['gpu_name'] = torch.cuda.get_device_name(0)
                info['vram_total'] = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
                info['vram_allocated'] = f"{torch.cuda.memory_allocated(0) / (1024**3):.2f} GB"
            except:
                pass
        
        return info
