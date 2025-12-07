"""
Wake Word Data Collection Tool
Records audio samples for training
"""

import os
import time
import wave
import pyaudio
import numpy as np
from pathlib import Path


class DataCollector:
    """
    Interactive tool to record wake word samples
    """
    
    def __init__(
        self,
        output_dir: str = "data",
        sample_rate: int = 16000,
        duration: float = 1.5,  # 1.5 seconds per sample
        channels: int = 1
    ):
        """
        Initialize data collector
        
        Args:
            output_dir: Where to save recordings
            sample_rate: 16kHz for speech
            duration: Length of each recording
            channels: Mono audio
        """
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.channels = channels
        self.chunk_size = 1024
        
        # Create directories
        self.wake_word_dir = self.output_dir / "wake_word"
        self.negative_dir = self.output_dir / "negative"
        self.wake_word_dir.mkdir(parents=True, exist_ok=True)
        self.negative_dir.mkdir(parents=True, exist_ok=True)
        
        # PyAudio setup
        self.audio = pyaudio.PyAudio()
    
    def record_sample(self, label: str) -> str:
        """
        Record one audio sample
        
        Args:
            label: 'wake_word' or 'negative'
        
        Returns:
            Path to saved file
        """
        print("\n🎤 Recording in 3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        print("🔴 RECORDING NOW!")
        
        # Open audio stream
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        # Record audio
        frames = []
        num_chunks = int(self.sample_rate / self.chunk_size * self.duration)
        
        for _ in range(num_chunks):
            data = stream.read(self.chunk_size)
            frames.append(data)
        
        print("✅ Recording complete!")
        
        # Stop stream
        stream.stop_stream()
        stream.close()
        
        # Save to file
        save_dir = self.wake_word_dir if label == "wake_word" else self.negative_dir
        timestamp = int(time.time() * 1000)
        filename = save_dir / f"{label}_{timestamp}.wav"
        
        with wave.open(str(filename), 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
        
        print(f"💾 Saved: {filename.name}")
        return str(filename)
    
    def collect_wake_word_samples(self, num_samples: int = 100):
        """
        Collect positive samples (you saying "Hey Zenith")
        
        Args:
            num_samples: How many samples to record
        """
        print("\n" + "=" * 60)
        print("🎯 WAKE WORD DATA COLLECTION")
        print("=" * 60)
        print(f"\n📝 Instructions:")
        print(f"   - Say 'Hey Zenith' clearly when recording starts")
        print(f"   - Vary your tone, speed, and volume")
        print(f"   - Record in different locations/noise levels")
        print(f"   - Target: {num_samples} samples")
        print("\n" + "=" * 60)
        
        for i in range(num_samples):
            print(f"\n📊 Sample {i+1}/{num_samples}")
            input("Press ENTER when ready...")
            self.record_sample("wake_word")
        
        print("\n✅ Wake word collection complete!")
    
    def collect_negative_samples(self, num_samples: int = 100):
        """
        Collect negative samples (anything that's NOT "Hey Zenith")
        
        Args:
            num_samples: How many samples to record
        """
        print("\n" + "=" * 60)
        print("🎯 NEGATIVE SAMPLE COLLECTION")
        print("=" * 60)
        print(f"\n📝 Instructions:")
        print(f"   - Say random words, sentences, cough, etc.")
        print(f"   - DO NOT say 'Hey Zenith'")
        print(f"   - Include background noise, music, TV")
        print(f"   - Target: {num_samples} samples")
        print("\n" + "=" * 60)
        
        for i in range(num_samples):
            print(f"\n📊 Sample {i+1}/{num_samples}")
            input("Press ENTER when ready...")
            self.record_sample("negative")
        
        print("\n✅ Negative sample collection complete!")
    
    def show_stats(self):
        """
        Display collected data statistics
        """
        wake_word_count = len(list(self.wake_word_dir.glob("*.wav")))
        negative_count = len(list(self.negative_dir.glob("*.wav")))
        
        print("\n" + "=" * 60)
        print("📊 DATA COLLECTION STATS")
        print("=" * 60)
        print(f"   ✅ Wake word samples: {wake_word_count}")
        print(f"   ❌ Negative samples: {negative_count}")
        print(f"   📁 Total samples: {wake_word_count + negative_count}")
        print("=" * 60)
    
    def cleanup(self):
        """Close PyAudio"""
        self.audio.terminate()


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    collector = DataCollector(output_dir="data")
    
    try:
        while True:
            print("\n" + "=" * 60)
            print("🎙️  WAKE WORD DATA COLLECTOR")
            print("=" * 60)
            print("\n1. Collect wake word samples ('Hey Zenith')")
            print("2. Collect negative samples (other speech/noise)")
            print("3. Show statistics")
            print("4. Exit")
            
            choice = input("\nChoice (1-4): ").strip()
            
            if choice == "1":
                num = int(input("How many samples? (recommended: 100-200): "))
                collector.collect_wake_word_samples(num)
            elif choice == "2":
                num = int(input("How many samples? (recommended: 200-300): "))
                collector.collect_negative_samples(num)
            elif choice == "3":
                collector.show_stats()
            elif choice == "4":
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice")
    
    finally:
        collector.cleanup()
