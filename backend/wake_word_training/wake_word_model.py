"""
Wake Word Detection CNN Model
Lightweight architecture for "Hey Zenith" detection
Based on best practices from research [web:28]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WakeWordCNN(nn.Module):
    """
    Convolutional Neural Network for wake word detection
    
    Architecture:
        - Input: MFCC features (39 x 100) = (channels x time_steps)
        - 3 Convolutional blocks with batch norm + dropout
        - Global average pooling
        - Fully connected classifier
        - Output: Binary (wake_word=1, not_wake_word=0)
    
    Designed for:
        - Low latency (<50ms inference) [web:53]
        - Small model size (~500KB)
        - CPU-friendly (works on your laptop)
        - 90%+ accuracy [web:28]
    """
    
    def __init__(self, dropout_rate: float = 0.3):
        """
        Initialize wake word CNN
        
        Args:
            dropout_rate: Dropout probability (0.3 = 30% dropout)
        """
        super(WakeWordCNN, self).__init__()
        
        # ============================================
        # CONV BLOCK 1: Extract low-level features
        # ============================================
        self.conv1 = nn.Conv2d(
            in_channels=1,      # Input: 1 channel (grayscale-like MFCC)
            out_channels=32,    # Learn 32 different features
            kernel_size=(3, 3), # 3x3 filter
            stride=1,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # ============================================
        # CONV BLOCK 2: Extract mid-level features
        # ============================================
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # ============================================
        # CONV BLOCK 3: Extract high-level features
        # ============================================
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            stride=1,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        # ============================================
        # GLOBAL AVERAGE POOLING
        # Reduces feature maps to single values
        # ============================================
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ============================================
        # FULLY CONNECTED CLASSIFIER
        # ============================================
        self.fc1 = nn.Linear(128, 64)
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 1)  # Binary output (sigmoid activation)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor (batch_size, 1, 39, 100)
               - 39 = MFCC features (13 base + 13 delta + 13 delta2)
               - 100 = time steps (fixed length)
        
        Returns:
            Output tensor (batch_size, 1) with probabilities
        """
        # ===== CONV BLOCK 1 =====
        x = self.conv1(x)           # (batch, 32, 39, 100)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)            # (batch, 32, 19, 50)
        x = self.dropout1(x)
        
        # ===== CONV BLOCK 2 =====
        x = self.conv2(x)            # (batch, 64, 19, 50)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)            # (batch, 64, 9, 25)
        x = self.dropout2(x)
        
        # ===== CONV BLOCK 3 =====
        x = self.conv3(x)            # (batch, 128, 9, 25)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)            # (batch, 128, 4, 12)
        x = self.dropout3(x)
        
        # ===== GLOBAL POOLING =====
        x = self.global_pool(x)      # (batch, 128, 1, 1)
        x = x.view(x.size(0), -1)    # Flatten: (batch, 128)
        
        # ===== CLASSIFIER =====
        x = self.fc1(x)              # (batch, 64)
        x = F.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)              # (batch, 1)
        x = torch.sigmoid(x)         # Convert to probability [0, 1]
        
        return x
    
    def count_parameters(self):
        """
        Count total trainable parameters
        
        Returns:
            Number of parameters (should be ~100k-500k for efficiency)
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================
# MODEL FACTORY FUNCTION
# ============================================

def create_wake_word_model(dropout_rate: float = 0.3):
    """
    Create and initialize wake word detection model
    
    Args:
        dropout_rate: Dropout probability for regularization
    
    Returns:
        Initialized WakeWordCNN model
    """
    model = WakeWordCNN(dropout_rate=dropout_rate)
    
    # Print model summary
    total_params = model.count_parameters()
    print(f"🧠 Wake Word CNN Model Created")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024:.1f} KB (float32)")
    print(f"   Dropout rate: {dropout_rate}")
    
    return model


# ============================================
# TEST MODEL (if run directly)
# ============================================

if __name__ == "__main__":
    # Create model
    model = create_wake_word_model(dropout_rate=0.3)
    
    # Test with dummy input
    dummy_input = torch.randn(4, 1, 39, 100)  # Batch of 4 samples
    print(f"\n🧪 Testing model with input shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"✅ Output shape: {output.shape}")
    print(f"   Output values: {output.squeeze().detach().numpy()}")
    print(f"   (Values should be between 0 and 1)")
