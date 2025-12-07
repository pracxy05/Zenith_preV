"""
Core Assistant Module - Production Grade v5.2
✅ Edge-TTS Integration
✅ Proper Barge-in Interrupt (TTS + LLM)
✅ All existing features preserved
"""

import pyaudio
import numpy as np
import time
import logging
import yaml
from typing import List, Dict, Any, Optional
from enum import Enum

from .wake_word import WakeWordDetector
from .speech import SpeechRecognizer
from .llm import LLMEngine
from .behavior import BehaviorPredictor
from .function_registry import FunctionRegistry
from .audio_processor import AudioProcessor
from .ffmpeg_audio import FFmpegAudioSource

# ✅ NEW: Import Edge-TTS instead of old TTS
from .tts_edge import EdgeTTS

logger = logging.getLogger(__name__)


class AssistantState(Enum):
    """Assistant state machine"""
    IDLE = "idle"
    WAKE_DETECTED = "wake_detected"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    CONTINUOUS = "continuous"


class ZenithAssistant:
    """
    Professional AI assistant with:
    - "Yes Boss" wake word acknowledgment
    - Edge-TTS natural voice
    - Proper barge-in interrupt (TTS + LLM)
    - Centralized tool and command registry
    - Behavior learning
    - Proper cleanup
    """
    
    def __init__(self, config_path: str = "config/settings.yaml") -> None:
        """Initialize Zenith Assistant with all components"""
        logger.info("🚀 Initializing Zenith Assistant v5.2 (Edge-TTS Upgrade)...")
        
        try:
            # Load configuration
            self.config = self._load_config(config_path)
            
            # Audio settings
            audio_config = self.config['audio']
            self.sample_rate = audio_config['sample_rate']
            self.chunk_size = audio_config['chunk_size']
            self.channels = audio_config['channels']
            self.use_ffmpeg = audio_config.get('use_ffmpeg', False)
            self.barge_in_enabled = audio_config.get('barge_in_enabled', True)
            
            # ✅ NEW: Barge-in settings
            self.barge_in_force_stop = audio_config.get('barge_in_force_stop', True)
            self.barge_in_clear_queue = audio_config.get('barge_in_clear_queue', True)
            
            # Initialize core components
            print("📦 Loading components...")
            self.wake_word = WakeWordDetector(self.config)
            self.speech = SpeechRecognizer(self.config)
            from tools.system import set_speech_recognizer
            set_speech_recognizer(self.speech)
            self.llm = LLMEngine(self.config)
            self.behavior = BehaviorPredictor(self.config)
            self.function_registry = FunctionRegistry()
            self.audio_processor = AudioProcessor(self.config)
            
            # ✅ NEW: Initialize Edge-TTS instead of old TTS
            self.tts = EdgeTTS(self.config.get('tts', {}))
            
            # Link TTS with audio processor
            self.tts.audio_processor = self.audio_processor
            
            # Register all tools
            from tools.tool_registry import register_all_tools
            register_all_tools(self.function_registry)
            
            # Initialize command registry
            from tools.command_registry import create_command_registry
            self.command_registry = create_command_registry(self.function_registry)
            
            # Audio source
            if self.use_ffmpeg:
                self.ffmpeg_source = FFmpegAudioSource(self.config)
                self.audio = None
                self.stream = None
                print("🎤 Audio Input: FFmpeg (Advanced filtering enabled)")
            else:
                self.ffmpeg_source = None
                self.audio = pyaudio.PyAudio()
                self.stream = None
                print("🎤 Audio Input: PyAudio (Standard)")
            
            # State management
            self.state = AssistantState.IDLE
            self.running = False
            self.conversation_history: List[Dict[str, str]] = []
            
            # Continuous mode
            self.continuous_mode = audio_config.get('continuous_mode', True)
            self.max_continuous_time = audio_config.get('max_continuous_time', 300)
            self.last_interaction_time = time.time()
            
            # Performance tracking
            self.interaction_count = 0
            self.successful_interactions = 0
            self.failed_interactions = 0
            self.start_time = time.time()
            
            # ✅ NEW: System settings
            system_config = self.config.get('system', {})
            self.wake_response_enabled = system_config.get('wake_response_enabled', True)
            self.wake_response_text = system_config.get('wake_response_text', 'Yes boss')
            self.wake_response_interrupt = system_config.get('wake_response_interrupt_current', True)
            
            print("✅ Initialization complete!\n")
            print("="*60)
            print("🎯 ZENITH v5.2 - Ready for Command")
            print("="*60)
            print(f"🔹 Continuous Mode: {'ON' if self.continuous_mode else 'OFF'}")
            print(f"🔹 Barge-in: {'ENABLED' if self.barge_in_enabled else 'DISABLED'}")
            print(f"🔹 Audio Source: {'FFmpeg' if self.use_ffmpeg else 'PyAudio'}")
            print(f"🔹 LLM: {self.config['llm']['model']}")
            print(f"🔹 TTS: Edge-TTS (Microsoft Neural)")
            print(f"🔹 Voice: {self.config.get('tts', {}).get('edge_voice', 'en-US-GuyNeural')}")
            print(f"🔹 Wake Response: '{self.wake_response_text}' enabled")
            print("="*60 + "\n")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            raise
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Loaded config from {config_path}")
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load config from '{config_path}': {e}")
    
    def _check_audio_quality(self, audio_data: np.ndarray) -> bool:
        """Validate audio quality"""
        try:
            if audio_data is None or len(audio_data) == 0:
                return False
            
            rms = np.sqrt(np.mean(audio_data.astype(np.float64)**2))
            
            if rms < 100:
                logger.debug(f"Audio too quiet: RMS {rms:.1f}")
                print(f"❌ Audio too quiet (RMS: {rms:.0f})")
                return False
            
            if rms > 8000:
                logger.warning(f"Audio very loud: RMS {rms:.1f}")
                print(f"⚠️  Audio very loud (RMS: {rms:.0f})")
            
            peak = np.max(np.abs(audio_data))
            if peak > 30000:
                logger.warning(f"Audio clipping: peak {peak}")
                clipping_ratio = np.sum(np.abs(audio_data) > 30000) / len(audio_data)
                if clipping_ratio > 0.05:
                    print(f"❌ Severe distortion ({clipping_ratio*100:.1f}%)")
                    return False
            
            logger.debug(f"✅ Audio OK: RMS {rms:.1f}, Peak {peak}")
            return True
            
        except Exception as e:
            logger.error(f"Audio quality check error: {e}")
            return True
    
    def _say_yes_boss(self) -> None:
        """
        Say "Yes Boss" when wake word is detected
        ✅ FIXED: Proper interrupt with Edge-TTS
        """
        try:
            # Always print to console
            print(f"🤖 Zenith: {self.wake_response_text}")
            
            # Speak if TTS is enabled and wake response is enabled
            if self.tts and self.tts.enabled and self.wake_response_enabled:
                # ✅ FIXED: Interrupt current speech if enabled
                self.tts.speak(
                    self.wake_response_text,
                    interrupt_current=self.wake_response_interrupt
                )
            else:
                logger.debug("TTS disabled, skipping voice acknowledgment")
                
        except Exception as e:
            logger.error(f"Yes boss TTS error: {e}")
            print("   (TTS failed but continuing...)")
    
    def _force_stop_all_output(self) -> None:
        """
        ✅ NEW: Force stop ALL output (TTS + LLM) immediately
        Called when wake word detected during speaking
        """
        try:
            # Stop TTS immediately
            if self.tts and self.tts.is_speaking:
                self.tts.stop()
                logger.info("🛑 TTS force stopped")
            
            # Interrupt LLM generation
            if self.llm:
                self.llm.interrupt()
                logger.info("🛑 LLM generation interrupted")
            
        except Exception as e:
            logger.error(f"Force stop error: {e}")
    
    def listen(self) -> None:
        """Main listening loop with state machine"""
        try:
            # Start audio source
            if self.ffmpeg_source:
                self.ffmpeg_source.start()
            else:
                self.stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size
                )
            
            self.running = True
            self.state = AssistantState.IDLE
            
            print("\n" + "="*60)
            print("👂 LISTENING FOR 'HEY ZENITH'")
            print("="*60)
            if self.continuous_mode:
                print("💡 Continuous mode: Say 'stop' or interrupt anytime")
            if self.barge_in_enabled:
                print("🔥 Barge-in enabled: Say 'Hey Zenith' to interrupt")
            print(f"🎤 Say 'Hey Zenith' → I'll respond '{self.wake_response_text}'")
            print("="*60 + "\n")
            
            # Buffers
            audio_buffer = np.array([], dtype=np.int16)
            silence_frames = 0
            max_silence = self.config['audio']['silence_frames']
            speech_buffer = []
            speech_detected = False
            
            while self.running:
                try:
                    # Get audio chunk
                    if self.ffmpeg_source:
                        audio_chunk = self.ffmpeg_source.read_chunk()
                        if audio_chunk is None:
                            if not self.ffmpeg_source.is_active():
                                break
                            continue
                    else:
                        in_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                        audio_chunk = np.frombuffer(in_data, dtype=np.int16)
                    
                    # Audio processing
                    processed_audio = self.audio_processor.enhance_user_voice(audio_chunk)
                    self.audio_processor.update_threshold(processed_audio)
                    should_process = self.audio_processor.should_process_audio()
                    
                    # ✅ FIXED: Wake-word barge-in with force stop
                    if not should_process and self.barge_in_enabled:
                        if self.tts and self.tts.is_speaking:
                            # Check for wake word during TTS playback
                            if self.wake_word.process(processed_audio):
                                print("\n" + "="*60)
                                print("🎙️  BARGE-IN: Wake word detected!")
                                print("="*60 + "\n")
                                should_process = True
                                
                                # ✅ FIXED: Force stop everything
                                if self.barge_in_force_stop:
                                    self._force_stop_all_output()
                                else:
                                    # Legacy behavior (just stop TTS)
                                    if self.tts:
                                        self.tts.interrupt()
                                    if self.llm:
                                        self.llm.interrupt()
                                
                                # Say "Yes Boss"
                                self._say_yes_boss()
                                print("="*60 + "\n")
                                
                                self.state = AssistantState.LISTENING
                                speech_buffer = []
                                silence_frames = 0
                                speech_detected = False
                                print("🎤 Listening...\n")
                                continue
                            
                    if not should_process:
                        continue
                    
                    # State machine
                    if self.state in [AssistantState.IDLE, AssistantState.CONTINUOUS]:
                        audio_buffer = np.append(audio_buffer, processed_audio)
                        if len(audio_buffer) > self.sample_rate * 3:
                            audio_buffer = audio_buffer[-self.sample_rate * 3:]
                        
                        # ✅ FIXED: Changed .detect() to .process()
                        if self.wake_word.process(processed_audio):
                            print("\n" + "="*60)
                            print("🔥 WAKE WORD DETECTED!")
                            print("="*60)
                            
                            # ✅ Say "Yes Boss"
                            self._say_yes_boss()
                            print("="*60 + "\n")
                            
                            self.state = AssistantState.LISTENING
                            self.last_interaction_time = time.time()
                            speech_buffer = []
                            silence_frames = 0
                            speech_detected = False
                            print("🎤 Listening...\n")
                    
                    elif self.state == AssistantState.LISTENING:
                        volume = np.abs(processed_audio).mean()
                        threshold = self.audio_processor.current_threshold
                        
                        if volume > threshold:
                            if not speech_detected:
                                print("🗣️  Speech detected...")
                                speech_detected = True
                            speech_buffer.append(processed_audio)
                            silence_frames = 0
                        else:
                            if speech_detected:
                                silence_frames += 1
                            if speech_detected and silence_frames < max_silence:
                                speech_buffer.append(processed_audio)
                        
                        # End of speech
                        if silence_frames >= max_silence and speech_detected:
                            if len(speech_buffer) > 0:
                                audio_data = np.concatenate(speech_buffer)
                                min_length = int(self.config['audio']['min_speech_length'] * self.sample_rate)
                                
                                if len(audio_data) < min_length:
                                    print("❌ Speech too short\n")
                                else:
                                    self.state = AssistantState.PROCESSING
                                    self._process_speech(audio_data)
                            
                            speech_buffer = []
                            silence_frames = 0
                            speech_detected = False
                            
                            if self.continuous_mode:
                                self.state = AssistantState.CONTINUOUS
                                print("🎤 Ready for next command...\n")
                            else:
                                self.state = AssistantState.IDLE
                                print("\n👂 Listening for 'hey zenith'...\n")
                    
                    # Timeout
                    if self.continuous_mode and self.state == AssistantState.CONTINUOUS:
                        if time.time() - self.last_interaction_time > self.max_continuous_time:
                            print("💤 Timeout - returning to idle\n")
                            self.state = AssistantState.IDLE
                
                except KeyboardInterrupt:
                    print("\n\n⚠️  Keyboard interrupt detected (Ctrl+C)...")
                    break
                except Exception as e:
                    logger.error(f"Listen loop error: {e}", exc_info=True)
                    time.sleep(0.1)
        
        finally:
            self.cleanup()
    
    def _process_speech(self, audio_data: np.ndarray) -> None:
        """Process recorded speech"""
        try:
            # Quality check
            if not self._check_audio_quality(audio_data):
                self.failed_interactions += 1
                return
            
            # Transcribe
            user_text = self.speech.transcribe(audio_data)
            if not user_text:
                print("❌ Could not understand speech\n")
                self.failed_interactions += 1
                return
            
            print(f"👤 You: {user_text}\n")
            self.interaction_count += 1
            
            # Stop command
            if "stop" in user_text.lower() and len(user_text.split()) <= 3:
                print("💤 Going to sleep...\n")
                self.state = AssistantState.IDLE
                return
            
            # Process intent
            self.state = AssistantState.RESPONDING
            response = self._process_user_intent(user_text)
            
            # Update history
            self.conversation_history.append({"role": "user", "content": user_text})
            self.conversation_history.append({"role": "assistant", "content": response})
            max_history = self.config['llm']['max_history'] * 2
            if len(self.conversation_history) > max_history:
                self.conversation_history = self.conversation_history[-max_history:]
            
            # Behavior learning
            if self.behavior.enabled:
                self.behavior.log_interaction(user_text, response, success=True)
            
            self.successful_interactions += 1
            self.last_interaction_time = time.time()
            
        except Exception as e:
            logger.error(f"Speech processing error: {e}", exc_info=True)
            print(f"❌ Error: {e}\n")
            self.failed_interactions += 1
    
    def _process_user_intent(self, user_text: str) -> str:
        """Process user intent with command registry"""
        # Try command registry first
        command_result = self.command_registry.process_command(user_text)
        
        if command_result:
            function_name, kwargs = command_result
            return self._execute_function(function_name, **kwargs)
        
        # Fallback to LLM
        try:
            response = self.llm.generate_response(
                user_text,
                self.conversation_history,
                self.function_registry,
                tts_callback=self.tts.speak if self.config['llm'].get('streaming') else None
            )
            
            if not self.config['llm'].get('streaming'):
                print(f"\n🤖 Zenith: {response}\n")
                if self.tts and self.tts.enabled:
                    self.tts.speak(response)
                    # ✅ FIXED: Better wait loop with interrupt check
                    timeout = 30
                    elapsed = 0
                    while self.tts.is_speaking and elapsed < timeout:
                        time.sleep(0.05)
                        elapsed += 0.05
                        # Allow interrupt during wait
                        if not self.running:
                            break
            else:
                print()
            
            return response
            
        except Exception as e:
            logger.error(f"LLM error: {e}", exc_info=True)
            error_msg = "I encountered an error."
            if self.tts and self.tts.enabled:
                self.tts.speak(error_msg)
                timeout = 5
                elapsed = 0
                while self.tts.is_speaking and elapsed < timeout:
                    time.sleep(0.05)
                    elapsed += 0.05
            return error_msg
    
    def _execute_function(self, function_name: str, **kwargs) -> str:
        """Execute function with TTS"""
        try:
            logger.info(f"Executing: {function_name}({kwargs})")
            result = self.function_registry.execute(function_name, **kwargs)
            
            if isinstance(result, dict):
                response = "\n".join([f"{k}: {v}" for k, v in result.items()])
            else:
                response = str(result)
            
            print(f"🤖 Zenith: {response}\n")
            
            if self.tts and self.tts.enabled:
                self.tts.speak(response)
                timeout = 30
                elapsed = 0
                while self.tts.is_speaking and elapsed < timeout:
                    time.sleep(0.05)
                    elapsed += 0.05
                    # Allow interrupt during wait
                    if not self.running:
                        break
            
            return response
            
        except Exception as e:
            logger.error(f"Function error: {e}", exc_info=True)
            error_msg = f"Error: {str(e)}"
            print(f"❌ {error_msg}\n")
            if self.tts and self.tts.enabled:
                self.tts.speak(error_msg)
                timeout = 5
                elapsed = 0
                while self.tts.is_speaking and elapsed < timeout:
                    time.sleep(0.05)
                    elapsed += 0.05
            return error_msg
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        uptime = time.time() - self.start_time
        success_rate = 0.0
        if self.interaction_count > 0:
            success_rate = (self.successful_interactions / self.interaction_count) * 100
        
        return {
            'uptime_seconds': uptime,
            'uptime_formatted': f"{int(uptime // 60)}m {int(uptime % 60)}s",
            'total_interactions': self.interaction_count,
            'successful_interactions': self.successful_interactions,
            'failed_interactions': self.failed_interactions,
            'success_rate': f"{success_rate:.1f}%",
            'interactions_per_minute': (self.interaction_count / uptime) * 60 if uptime > 0 else 0,
            'conversation_length': len(self.conversation_history),
            'current_state': self.state.value,
            'behavior_stats': self.behavior.get_stats() if self.behavior.enabled else None
        }
    
    def cleanup(self) -> None:
        """Clean up resources"""
        print("\n" + "="*60)
        print("🛑 SHUTTING DOWN ZENITH")
        print("="*60 + "\n")
        
        self.running = False
        
        # Close browser if open
        try:
            from tools.web_control import get_web_controller
            web_controller = get_web_controller()
            if web_controller and web_controller.is_browser_open:
                print("🌐 Closing browser...")
                web_controller.close_browser()
        except Exception as e:
            logger.error(f"Browser cleanup error: {e}")
        
        # Stop audio
        if self.ffmpeg_source:
            try:
                self.ffmpeg_source.stop()
            except Exception as e:
                logger.error(f"FFmpeg cleanup error: {e}")
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"Stream cleanup error: {e}")
        
        if self.audio:
            try:
                self.audio.terminate()
            except Exception as e:
                logger.error(f"PyAudio cleanup error: {e}")
        
        # ✅ NEW: Stop TTS (Edge-TTS cleanup)
        if self.tts:
            try:
                self.tts.cleanup()
            except Exception as e:
                logger.error(f"TTS cleanup error: {e}")
        
        # Save behavior
        if self.behavior and self.behavior.enabled:
            try:
                self.behavior.save_model()
                print("💾 Behavior model saved")
            except Exception as e:
                logger.error(f"Behavior save error: {e}")
        
        # Print stats
        stats = self.get_stats()
        print("\n📊 SESSION STATISTICS")
        print("="*60)
        print(f"⏱️  Uptime: {stats['uptime_formatted']}")
        print(f"💬 Total Interactions: {stats['total_interactions']}")
        print(f"✅ Successful: {stats['successful_interactions']}")
        print(f"❌ Failed: {stats['failed_interactions']}")
        print(f"📈 Success Rate: {stats['success_rate']}")
        print(f"🔄 Interactions/min: {stats['interactions_per_minute']:.2f}")
        print("="*60)
        print("\n✅ Cleanup complete - Goodbye!\n")
