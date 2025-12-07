"""
Text-to-Speech Module - Streaming with Barge-in Support
✅ FIX #4: Better error handling to prevent crashes
Sentence-by-sentence playback with interrupt capability
"""

import logging
import threading
import subprocess
import tempfile
import os
import time
import queue
import re
from typing import Optional

import pygame

logger = logging.getLogger(__name__)

class TextToSpeech:
    """TTS engine with streaming and interrupt support"""
    
    def __init__(self, config: dict):
        """Initialize TTS engine"""
        self.config = config
        self.tts_config = config.get('tts', {})
        self.enabled = self.tts_config.get('enabled', True)
        self.engine_type = self.tts_config.get('engine', 'piper')
        self.streaming = self.tts_config.get('streaming', False)
        self.min_sentence_length = self.tts_config.get('min_sentence_length', 10)
        
        # State control
        self.is_speaking = False
        self.should_stop = False
        self.sentence_queue: queue.Queue = queue.Queue()
        
        # Audio processor reference
        self.audio_processor = None
        
        # ✅ FIX #4A: Better pygame initialization with fallback
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            self.use_pygame = True
            logger.info("✅ Pygame mixer initialized")
        except Exception as e:
            logger.warning(f"⚠️  Pygame mixer failed: {e}")
            self.use_pygame = False
            self.enabled = False
            return
        
        if not self.enabled:
            print("🔊 TTS disabled")
            return
        
        try:
            if self.engine_type == 'piper':
                self.engine = self._init_piper()
                model_name = os.path.basename(self.engine['model_path']).replace('.onnx', '')
                print(f"✅ Voice model ready: {model_name}")
            else:
                raise ValueError(f"Unknown TTS engine: {self.engine_type}")
                
        except Exception as e:
            logger.error(f"TTS init error: {e}")
            print(f"❌ TTS initialization failed: {e}")
            self.enabled = False
            return
        
        # Start TTS worker thread
        self.worker_thread = threading.Thread(
            target=self._tts_worker,
            daemon=True,
            name="TTSWorker"
        )
        self.worker_thread.start()
    
    def _init_piper(self) -> dict:
        """Initialize Piper TTS with flexible model path detection"""
        model = self.tts_config.get('model', 'en_US-lessac-medium')
        piper_path = self.tts_config.get('piper_path', 'piper/piper.exe')
        
        if not os.path.exists(piper_path):
            raise FileNotFoundError(f"Piper not found: {piper_path}")
        
        # Try multiple model path locations
        model_base_path = self.tts_config.get('model_base_path', 'piper/models')
        
        # Check different possible locations
        possible_paths = [
            os.path.join(model_base_path, f'{model}.onnx'),
            os.path.join('piper', 'models', f'{model}.onnx'),
            os.path.join('models', 'piper_voices', f'{model}.onnx'),
            os.path.join('piper_voices', f'{model}.onnx'),
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"✅ Found model at: {path}")
                break
        
        if not model_path:
            error_msg = f"Model '{model}.onnx' not found in any location. Tried:\n"
            error_msg += "\n".join(f"  - {p}" for p in possible_paths)
            raise FileNotFoundError(error_msg)
        
        return {
            'piper_path': piper_path,
            'model_path': model_path,
            'speed': self.tts_config.get('speed', 1.0)
        }
    
    def _tts_worker(self) -> None:
        """Background worker that processes TTS queue"""
        while True:
            try:
                # Get next sentence (blocking)
                sentence = self.sentence_queue.get(timeout=0.1)
                
                if sentence == "STOP":
                    # Flush queue
                    while not self.sentence_queue.empty():
                        try:
                            self.sentence_queue.get_nowait()
                        except:
                            break
                    self.is_speaking = False
                    continue
                
                # Speak this sentence
                self._speak_sentence(sentence)
                
            except queue.Empty:
                continue
            except Exception as e:
                # ✅ FIX #4B: Don't crash worker thread on errors
                logger.error(f"TTS worker error: {e}")
                self.is_speaking = False
    
    def _speak_sentence(self, text: str) -> None:
        """Speak a single sentence"""
        if not self.enabled or not text.strip():
            return
        
        self.is_speaking = True
        
        # Notify audio processor
        if self.audio_processor:
            self.audio_processor.notify_tts_start()
        
        temp_file = None
        process = None
        
        try:
            # Generate audio file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_file = f.name
            
            # Piper command
            cmd = [
                self.engine['piper_path'],
                '--model', self.engine['model_path'],
                '--output_file', temp_file
            ]
            
            # ✅ FIX #4C: Increased timeout from 10s to 15s
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            
            try:
                stdout, stderr = process.communicate(input=text.encode('utf-8'), timeout=15.0)
                
                if process.returncode != 0:
                    error_msg = stderr.decode('utf-8') if stderr else "Unknown error"
                    logger.error(f"Piper generation failed: {error_msg}")
                    return
                    
            except subprocess.TimeoutExpired:
                # ✅ FIX #4D: Better timeout handling
                logger.error("Piper timeout (15s exceeded)")
                if process:
                    process.kill()
                    try:
                        process.wait(timeout=1.0)
                    except:
                        pass
                return
            
            # Play audio with interrupt checking
            if self.use_pygame and os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.play()
                
                # Wait for playback (check for interrupt every 50ms)
                while pygame.mixer.music.get_busy():
                    if self.should_stop:
                        pygame.mixer.music.stop()
                        logger.info("TTS playback interrupted")
                        break
                    time.sleep(0.05)
            else:
                logger.warning(f"Audio file not generated or empty: {temp_file}")
                
        except Exception as e:
            # ✅ FIX #4E: Log but don't crash
            logger.error(f"TTS speak error: {e}")
            
        finally:
            # Cleanup temp file
            if temp_file:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception as e:
                    logger.debug(f"Temp file cleanup failed: {e}")
            
            # Check if queue is empty
            if self.sentence_queue.empty():
                self.is_speaking = False
                if self.audio_processor:
                    self.audio_processor.notify_tts_end()
    
    def speak_streaming(self, text: str) -> None:
        """
        Queue text for streaming TTS (non-blocking)
        ✅ FIX #4F: Added safety checks
        """
        if not self.enabled or not text.strip():
            return
        
        try:
            # If streaming disabled, queue full text
            if not self.streaming:
                self.sentence_queue.put(text)
                return
            
            # Split into sentences
            sentences = re.split(r'([.!?\n]+\s*)', text)
            buffer = ""
            
            for part in sentences:
                buffer += part
                
                # Complete sentence detected
                if part.strip() and any(p in part for p in ['.', '!', '?', '\n']):
                    sentence = buffer.strip()
                    if len(sentence) >= self.min_sentence_length:
                        self.sentence_queue.put(sentence)
                    buffer = ""
            
            # Queue remaining text if significant
            if buffer.strip() and len(buffer.strip()) >= self.min_sentence_length:
                self.sentence_queue.put(buffer.strip())
                
        except Exception as e:
            # ✅ FIX #4G: Don't crash on queueing errors
            logger.error(f"TTS queueing error: {e}")
    
    def speak(self, text: str) -> None:
        """
        Speak full text (blocking - for compatibility)
        ✅ FIX #4H: Added safety and timeout
        """
        if not self.enabled or not text.strip():
            return
        
        try:
            self.speak_streaming(text)
            
            # Wait for queue to empty (with timeout)
            max_wait = 30  # 30 seconds max
            elapsed = 0
            
            while (not self.sentence_queue.empty() or self.is_speaking) and elapsed < max_wait:
                if self.should_stop:
                    break
                time.sleep(0.1)
                elapsed += 0.1
            
            if elapsed >= max_wait:
                logger.warning(f"TTS speak timeout after {max_wait}s")
                
        except Exception as e:
            # ✅ FIX #4I: Don't crash on speak errors
            logger.error(f"TTS speak error: {e}")
    
    def interrupt(self) -> None:
        """Interrupt current speech"""
        try:
            self.should_stop = True
            self.sentence_queue.put("STOP")  # Signal to flush
            
            if self.use_pygame:
                try:
                    pygame.mixer.music.stop()
                except:
                    pass
            
            time.sleep(0.2)
            self.should_stop = False
            logger.info("TTS interrupted")
            
        except Exception as e:
            # ✅ FIX #4J: Don't crash on interrupt errors
            logger.error(f"TTS interrupt error: {e}")
    
    def set_audio_processor(self, processor) -> None:
        """Set audio processor for coordination"""
        self.audio_processor = processor
    
    def is_queue_empty(self) -> bool:
        """Check if TTS queue is empty"""
        return self.sentence_queue.empty() and not self.is_speaking
