"""
LLM Engine Module - Streaming with Real-time TTS
Generates text chunk-by-chunk for sentence-by-sentence speech
"""

import ollama
import time
import logging
import re
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger(__name__)

class LLMEngine:
    """LLM engine with streaming and TTS callback support"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize LLM engine"""
        self.config = config
        self.model_name: str = config['llm']['model']
        self.system_prompt: str = config['llm']['system_prompt']
        self.timeout: float = config['llm'].get('timeout', 30.0)
        self.streaming: bool = config['llm'].get('streaming', False)
        
        # Interrupt control
        self.is_generating = False
        self.should_stop = False
        
        self._test_model_availability()
    
    def _test_model_availability(self) -> None:
        """Test if model is available"""
        try:
            print(f"🤖 Testing LLM connection ({self.model_name})...")
            ollama.list()
            print("✅ LLM connection established")
        except Exception as e:
            raise RuntimeError(f"LLM connection failed: {e}")
    
    def generate_response(
        self,
        user_text: str,
        history: List[Dict[str, str]],
        function_registry: Any,
        tts_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Generate response with optional streaming TTS"""
        self.is_generating = True
        self.should_stop = False
        
        try:
            # Build messages
            messages: List[Dict] = [
                {"role": "system", "content": self.system_prompt}
            ]
            messages.extend(history[-10:])  # Last 10 messages
            messages.append({"role": "user", "content": user_text})
            
            print("🤔 ", end="", flush=True)
            full_response: str = ""
            sentence_buffer: str = ""
            start_time = time.time()
            
            # Stream response
            for chunk in ollama.chat(
                model=self.model_name,
                messages=messages,
                stream=True
            ):
                # Check for interrupt
                if self.should_stop:
                    print("\n🛑 Generation interrupted")
                    break
                
                # Check for timeout
                if time.time() - start_time > self.timeout:
                    print("\n⏱️ Generation timeout")
                    break
                
                content: str = chunk['message']['content']
                full_response += content
                sentence_buffer += content
                
                print(content, end="", flush=True)
                
                # When sentence complete, send to TTS
                if self.streaming and tts_callback:
                    # Check for sentence endings
                    if any(p in sentence_buffer for p in ['.', '!', '?', '\n']):
                        sentence = sentence_buffer.strip()
                        if len(sentence) >= 10:  # Minimum length
                            tts_callback(sentence)
                            sentence_buffer = ""
            
            # Send remaining text to TTS
            if self.streaming and tts_callback and sentence_buffer.strip():
                if len(sentence_buffer.strip()) >= 10:
                    tts_callback(sentence_buffer.strip())
            
            print()  # Newline
            
            if not full_response.strip():
                return "I'm having trouble processing that."
            
            return full_response
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return "I encountered an error while processing your request."
        
        finally:
            self.is_generating = False
    
    def interrupt(self) -> None:
        """Interrupt ongoing generation"""
        self.should_stop = True
        logger.info("LLM generation interrupted")
