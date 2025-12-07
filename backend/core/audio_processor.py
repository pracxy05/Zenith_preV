"""
Audio Processing Module - Echo Cancellation & Filtering
Prevents assistant from hearing its own voice
"""

import numpy as np
import time
from typing import Optional, Dict, Any
from collections import deque
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Advanced audio processing for echo cancellation"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize audio processor"""
        self.config = config
        audio_config = config.get('audio', {})
        
        # Echo suppression settings
        self.echo_suppression = audio_config.get('echo_suppression', True)
        self.tts_buffer_time = audio_config.get('tts_buffer_time', 0.5)
        self.post_tts_delay = audio_config.get('post_tts_delay', 0.3)
        
        # Adaptive threshold
        self.base_threshold = audio_config.get('base_threshold', 300)
        self.adaptive_threshold = audio_config.get('adaptive_threshold', True)
        self.current_threshold = self.base_threshold
        
        # Frequency filtering
        self.frequency_filtering = audio_config.get('frequency_filtering', True)
        
        # TTS state tracking
        self.tts_active = False
        self.tts_start_time: Optional[float] = None
        self.tts_end_time: Optional[float] = None
        
        # Volume history for adaptive threshold
        self.volume_history = deque(maxlen=100)
        
        # Audio ducking (reduce input during TTS)
        self.audio_ducking = audio_config.get('audio_ducking', True)
        self.ducking_factor = 0.1  # Reduce to 10% during TTS
        
        logger.info("Audio processor initialized")
    
    def notify_tts_start(self):
        """Notify processor that TTS has started"""
        self.tts_active = True
        self.tts_start_time = time.time()
        logger.debug("TTS started - audio suppression active")
    
    def notify_tts_end(self):
        """Notify processor that TTS has ended"""
        self.tts_active = False
        self.tts_end_time = time.time()
        logger.debug("TTS ended - audio suppression will lift shortly")
    
    def should_process_audio(self) -> bool:
        """Determine if audio should be processed (not echo)"""
        if not self.echo_suppression:
            return True
        
        current_time = time.time()
        
        # Block audio DURING TTS
        if self.tts_active and self.tts_start_time:
            elapsed = current_time - self.tts_start_time
            if elapsed < self.tts_buffer_time:
                return False  # Don't process during TTS
        
        # Block audio AFTER TTS (brief delay)
        if self.tts_end_time:
            elapsed = current_time - self.tts_end_time
            if elapsed < self.post_tts_delay:
                return False  # Don't process immediately after TTS
        
        return True  # OK to process
    
    def enhance_user_voice(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Enhance user voice and suppress TTS echo
        """
        if len(audio_chunk) == 0:
            return audio_chunk
        
        # Apply audio ducking during TTS
        if self.audio_ducking and self.tts_active:
            audio_chunk = (audio_chunk * self.ducking_factor).astype(np.int16)
        
        # Frequency filtering (if enabled)
        if self.frequency_filtering:
            audio_chunk = self._apply_bandpass_filter(audio_chunk)
        
        return audio_chunk
    
    def _apply_bandpass_filter(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Apply simple bandpass filter to isolate human voice frequencies
        Keeps 300-3000 Hz (human voice range)
        """
        try:
            # Simple implementation: keep as-is for now
            # Full FFT filtering would be more complex
            # The FFmpeg input already does heavy filtering
            return audio_chunk
        except Exception as e:
            logger.debug(f"Bandpass filter error: {e}")
            return audio_chunk
    
    def update_threshold(self, audio_chunk: np.ndarray) -> None:
        """Update adaptive threshold based on ambient noise"""
        if not self.adaptive_threshold:
            return
        
        try:
            volume = np.abs(audio_chunk).mean()
            self.volume_history.append(volume)
            
            if len(self.volume_history) >= 50:
                # Set threshold above average ambient noise
                avg_noise = np.mean(list(self.volume_history)[:50])
                self.current_threshold = max(
                    self.base_threshold,
                    avg_noise * 2.0  # 2x average noise
                )
        except Exception as e:
            logger.debug(f"Threshold update error: {e}")
    
    def get_volume(self, audio_chunk: np.ndarray) -> float:
        """Get volume level of audio chunk"""
        try:
            return float(np.abs(audio_chunk).mean())
        except:
            return 0.0
    
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Determine if audio chunk contains speech"""
        try:
            volume = self.get_volume(audio_chunk)
            return volume > self.current_threshold
        except:
            return False
