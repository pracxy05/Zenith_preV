"""
FFmpeg Audio Input - Production Grade v4.1
Optimized volume boost to prevent clipping
"""

import subprocess
import threading
import queue
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class FFmpegAudioSource:
    """FFmpeg-based audio input with real-time filtering and balanced volume boost"""
    
    def __init__(self, config: dict):
        """Initialize FFmpeg audio source"""
        self.config = config
        audio_config = config.get('audio', {})
        
        self.sample_rate = audio_config.get('sample_rate', 16000)
        self.channels = audio_config.get('channels', 1)
        self.chunk_size = audio_config.get('chunk_size', 512)
        
        # FFmpeg process
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.audio_queue: queue.Queue = queue.Queue(maxsize=100)
        
        # Get microphone name from config
        self.device_name = audio_config.get('microphone_name', None)
        if not self.device_name:
            raise ValueError("microphone_name not set in config/settings.yaml")
        
        logger.info(f"FFmpeg audio source initialized: {self.device_name}")
    
    def start(self) -> None:
        """Start FFmpeg audio capture with optimized filters"""
        if self.is_running:
            return
        
        # ✅ OPTIMIZED FFmpeg command with BALANCED VOLUME BOOST
        cmd = [
            'ffmpeg',
            '-f', 'dshow',
            '-i', f'audio={self.device_name}',
            
            # ✅ OPTIMIZED: Volume boost + voice isolation filters
            '-af', (
                'highpass=f=200,'      # Remove low rumble (AC, traffic)
                'lowpass=f=3000,'      # Remove high hiss (electrical)
                'afftdn=nf=-25,'       # AI noise reduction
                'volume=1.5'           # ✅ 1.5x VOLUME BOOST (prevents clipping, was 2.0)
            ),
            
            # Force exact output format for Whisper
            '-f', 's16le',   # 16-bit signed PCM
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',      # Mono (1 channel)
            'pipe:1'         # Output to stdout
        ]
        
        # Start FFmpeg subprocess
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,  # Suppress FFmpeg logs
                bufsize=self.chunk_size * 2
            )
        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg not found. Install from https://ffmpeg.org/download.html "
                "and add to system PATH."
            )
        except Exception as e:
            raise RuntimeError(f"FFmpeg start failed: {e}")
        
        self.is_running = True
        
        # Start reader thread
        self.reader_thread = threading.Thread(
            target=self._read_audio,
            daemon=True,
            name="FFmpegReader"
        )
        self.reader_thread.start()
        
        print(f"🎤 FFmpeg audio capture started: {self.device_name} (16kHz mono, 1.5x boost)")
    
    def _read_audio(self) -> None:
        """Read audio from FFmpeg stdout in real-time"""
        bytes_per_chunk = self.chunk_size * 2  # 16-bit = 2 bytes per sample
        
        while self.is_running and self.process:
            try:
                # Read chunk from FFmpeg
                raw_data = self.process.stdout.read(bytes_per_chunk)
                
                if not raw_data or len(raw_data) < bytes_per_chunk:
                    logger.warning("FFmpeg stream ended")
                    break
                
                # Convert to numpy array (int16)
                audio_data = np.frombuffer(raw_data, dtype=np.int16)
                
                # Put in queue (non-blocking)
                try:
                    self.audio_queue.put_nowait(audio_data)
                except queue.Full:
                    # Drop oldest frame if queue full (prevent lag)
                    try:
                        self.audio_queue.get_nowait()
                        self.audio_queue.put_nowait(audio_data)
                    except:
                        pass
                        
            except Exception as e:
                if self.is_running:
                    logger.error(f"FFmpeg read error: {e}")
                break
    
    def read_chunk(self) -> Optional[np.ndarray]:
        """
        Read audio chunk from queue
        
        Returns:
            Audio data as int16 numpy array, or None if timeout
        """
        try:
            return self.audio_queue.get(timeout=1.0)
        except queue.Empty:
            return None
    
    def stop(self) -> None:
        """Stop FFmpeg capture and cleanup"""
        self.is_running = False
        
        if self.process:
            # Graceful shutdown
            self.process.terminate()
            
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                # Force kill if doesn't terminate
                logger.warning("FFmpeg didn't terminate gracefully, forcing kill")
                self.process.kill()
                self.process.wait()
            
            self.process = None
        
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break
        
        print("🎤 FFmpeg audio capture stopped")
    
    def is_active(self) -> bool:
        """Check if FFmpeg process is running"""
        return self.is_running and self.process and self.process.poll() is None
