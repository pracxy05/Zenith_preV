"""
Edge-TTS Engine - Natural Microsoft Neural Voices
100% free, no API key, works offline after first download
"""

import edge_tts
import asyncio
import pygame
import tempfile
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EdgeTTS:
    """
    Microsoft Edge TTS with natural neural voices
    """
    
    def __init__(self, config: dict):
        """
        Initialize Edge TTS
        
        Args:
            config: TTS configuration from settings.yaml
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        
        # Voice settings
        self.voice = config.get('edge_voice', 'en-US-GuyNeural')
        self.rate = config.get('rate', '+0%')  # Speed: -50% to +100%
        self.volume = config.get('volume', '+0%')  # Volume: -50% to +50%
        self.pitch = config.get('pitch', '+0Hz')  # Pitch: -50Hz to +50Hz
        
        # Pygame mixer for playback
        pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=2048)
        self.is_speaking = False
        self.current_task = None
        
        # Temp directory for audio files
        self.temp_dir = Path(tempfile.gettempdir()) / "zenith_tts"
        self.temp_dir.mkdir(exist_ok=True)
        
        logger.info("✅ Edge-TTS initialized")
        logger.info(f"   Voice: {self.voice}")
        logger.info(f"   Rate: {self.rate}, Volume: {self.volume}")
    
    def speak(self, text: str, interrupt_current: bool = True) -> None:
        """
        Speak text using Edge-TTS
        
        Args:
            text: Text to speak
            interrupt_current: Stop current speech if speaking
        """
        if not self.enabled or not text:
            return
        
        try:
            # Stop current speech if requested
            if interrupt_current and self.is_speaking:
                self.stop()
            
            # Run async TTS in sync context
            asyncio.run(self._speak_async(text))
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
    
    async def _speak_async(self, text: str) -> None:
        """
        Async TTS generation and playback
        """
        try:
            # Generate audio file
            audio_file = self.temp_dir / f"tts_{hash(text)}.mp3"
            
            # Create TTS communicate object
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.voice,
                rate=self.rate,
                volume=self.volume,
                pitch=self.pitch
            )
            
            # Save audio
            await communicate.save(str(audio_file))
            
            # Play audio
            self.is_speaking = True
            pygame.mixer.music.load(str(audio_file))
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
            
            self.is_speaking = False
            
            # Cleanup
            try:
                audio_file.unlink()
            except:
                pass
            
        except Exception as e:
            logger.error(f"TTS generation error: {e}")
            self.is_speaking = False
    
    def stop(self) -> None:
        """
        Stop current speech immediately
        """
        try:
            pygame.mixer.music.stop()
            self.is_speaking = False
            logger.info("🛑 TTS stopped")
        except Exception as e:
            logger.error(f"TTS stop error: {e}")
    
    def interrupt(self) -> None:
        """
        Interrupt current speech (alias for stop)
        """
        self.stop()
    
    def cleanup(self) -> None:
        """
        Cleanup resources
        """
        try:
            self.stop()
            pygame.mixer.quit()
            
            # Cleanup temp files
            for file in self.temp_dir.glob("tts_*.mp3"):
                try:
                    file.unlink()
                except:
                    pass
            
            logger.info("✅ Edge-TTS cleanup complete")
        except Exception as e:
            logger.error(f"TTS cleanup error: {e}")
    
    @staticmethod
    def list_voices() -> list:
        """
        Get list of available voices
        
        Returns:
            List of voice names
        """
        try:
            voices = asyncio.run(edge_tts.list_voices())
            return [v['ShortName'] for v in voices]
        except:
            return []
    
    @staticmethod
    def get_voice_info(voice_name: str = None) -> dict:
        """
        Get detailed voice information
        
        Args:
            voice_name: Voice to get info for (default: list all en-US)
        
        Returns:
            Voice details
        """
        try:
            voices = asyncio.run(edge_tts.list_voices())
            
            if voice_name:
                return next((v for v in voices if v['ShortName'] == voice_name), None)
            else:
                # Return all English (US) voices
                return [v for v in voices if v['Locale'].startswith('en-US')]
        except:
            return {}


# ============================================
# TEST FUNCTION
# ============================================

if __name__ == "__main__":
    # Test Edge-TTS
    import yaml
    
    config = {
        'enabled': True,
        'edge_voice': 'en-US-GuyNeural',
        'rate': '+0%',
        'volume': '+0%',
        'pitch': '+0Hz'
    }
    
    tts = EdgeTTS(config)
    
    print("\n🎤 Testing Edge-TTS...")
    print("="*60)
    
    # Test voice
    tts.speak("Yes boss, Zenith is ready with Edge TTS")
    
    print("\n✅ Test complete!")
    
    # List available voices
    print("\n📋 Available English (US) voices:")
    voices = EdgeTTS.get_voice_info()
    for voice in voices[:5]:  # Show first 5
        print(f"   - {voice['ShortName']}: {voice['Gender']}")
