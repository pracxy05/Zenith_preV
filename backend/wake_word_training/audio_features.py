"""
Audio Feature Extraction for Wake Word Detection
Extracts MFCC features from audio files
"""

import librosa
import numpy as np
import torch


class AudioFeatureExtractor:
    """
    Extract MFCC features from audio for wake word detection
    Based on best practices from research [web:28]
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,           # Standard MFCC coefficients
        n_fft: int = 512,            # FFT window size
        hop_length: int = 160,       # ~10ms at 16kHz
        win_length: int = 400,       # ~25ms at 16kHz
        n_mels: int = 40,            # Mel filter banks
        target_length: int = 100     # Fixed length for model input
    ):
        """
        Initialize feature extractor
        
        Args:
            sample_rate: Audio sample rate (16kHz recommended for speech)
            n_mfcc: Number of MFCC coefficients (13 is standard)
            n_fft: FFT size
            hop_length: Hop between frames (10ms is standard)
            win_length: Window length (25ms is standard)
            n_mels: Number of mel frequency bins
            target_length: Fixed output length (pad/trim to this)
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.target_length = target_length
    
    def extract_features(self, audio_path: str) -> np.ndarray:
        """
        Extract MFCC features from audio file
        
        Args:
            audio_path: Path to .wav file
        
        Returns:
            MFCC features (13 x target_length) normalized
        """
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Extract MFCC features [web:33]
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=0,
            fmax=sr / 2
        )
        
        # Add delta features (velocity of MFCC changes) [web:28]
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Stack all features (13 + 13 + 13 = 39 features)
        features = np.vstack([mfcc, delta, delta2])
        
        # Normalize to fixed length (pad or trim)
        features = self._normalize_length(features)
        
        # Standardize (zero mean, unit variance)
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        return features.astype(np.float32)
    
    def extract_features_from_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract features from raw audio array (for real-time detection)
        
        Args:
            audio: Raw audio numpy array
        
        Returns:
            MFCC features (39 x target_length)
        """
        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels
        )
        
        # Add deltas
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        features = np.vstack([mfcc, delta, delta2])
        features = self._normalize_length(features)
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        return features.astype(np.float32)
    
    def _normalize_length(self, features: np.ndarray) -> np.ndarray:
        """
        Pad or trim features to target length
        
        Args:
            features: Input features (n_features x time)
        
        Returns:
            Features with fixed time dimension
        """
        current_length = features.shape[1]
        
        if current_length < self.target_length:
            # Pad with zeros
            pad_width = self.target_length - current_length
            features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
        elif current_length > self.target_length:
            # Trim to target length
            features = features[:, :self.target_length]
        
        return features
