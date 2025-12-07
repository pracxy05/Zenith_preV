"""
Wake Word Model Training Script
Trains the CNN on collected "Hey Zenith" samples
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from audio_features import AudioFeatureExtractor
from wake_word_model import create_wake_word_model


class WakeWordDataset(Dataset):
    """
    PyTorch Dataset for wake word detection
    Loads audio files and extracts MFCC features [web:50]
    """
    
    def __init__(self, audio_files, labels, feature_extractor):
        """
        Initialize dataset
        
        Args:
            audio_files: List of audio file paths
            labels: List of labels (1=wake_word, 0=negative)
            feature_extractor: AudioFeatureExtractor instance
        """
        self.audio_files = audio_files
        self.labels = labels
        self.feature_extractor = feature_extractor
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        """
        Get one sample
        
        Returns:
            Tuple of (features, label)
        """
        # Extract MFCC features from audio file
        features = self.feature_extractor.extract_features(self.audio_files[idx])
        
        # Add channel dimension: (39, 100) -> (1, 39, 100)
        features = features[np.newaxis, :]
        
        # Convert to tensors
        features_tensor = torch.FloatTensor(features)
        label_tensor = torch.FloatTensor([self.labels[idx]])
        
        return features_tensor, label_tensor


def load_dataset(data_dir: str):
    """
    Load audio files and create labels [web:28]
    
    Args:
        data_dir: Path to data directory
    
    Returns:
        Tuple of (file_paths, labels)
    """
    data_path = Path(data_dir)
    
    # Load wake word samples (label = 1)
    wake_word_files = list((data_path / "wake_word").glob("*.wav"))
    wake_word_labels = [1] * len(wake_word_files)
    
    # Load negative samples (label = 0)
    negative_files = list((data_path / "negative").glob("*.wav"))
    negative_labels = [0] * len(negative_files)
    
    # Combine
    all_files = wake_word_files + negative_files
    all_labels = wake_word_labels + negative_labels
    
    print(f"📊 Dataset loaded:")
    print(f"   ✅ Wake word samples: {len(wake_word_files)}")
    print(f"   ❌ Negative samples: {len(negative_files)}")
    print(f"   📁 Total samples: {len(all_files)}")
    
    return all_files, all_labels


def train_model(
    data_dir: str = "data",
    output_dir: str = "../../models",
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    test_size: float = 0.2,
    val_size: float = 0.1
):
    """
    Train wake word detection model [web:54][web:56]
    
    Args:
        data_dir: Path to collected audio data
        output_dir: Where to save trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        test_size: Fraction of data for testing (0.2 = 20%)
        val_size: Fraction of training data for validation (0.1 = 10%)
    """
    print("\n" + "="*60)
    print("🎯 WAKE WORD MODEL TRAINING")
    print("="*60)
    
    # ============================================
    # STEP 1: Load and split dataset [web:54]
    # ============================================
    print("\n📂 Loading dataset...")
    all_files, all_labels = load_dataset(data_dir)
    
    # Split: 80% train, 20% test [web:54]
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, all_labels,
        test_size=test_size,
        random_state=42,
        stratify=all_labels  # Keep class balance [web:54]
    )
    
    # Split train into train/val: 72% train, 8% val, 20% test
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files, train_labels,
        test_size=val_size,
        random_state=42,
        stratify=train_labels
    )
    
    print(f"\n📊 Data split:")
    print(f"   🏋️ Training: {len(train_files)} samples ({len(train_files)/len(all_files)*100:.1f}%)")
    print(f"   ✅ Validation: {len(val_files)} samples ({len(val_files)/len(all_files)*100:.1f}%)")
    print(f"   🧪 Testing: {len(test_files)} samples ({len(test_files)/len(all_files)*100:.1f}%)")
    
    # ============================================
    # STEP 2: Create datasets and dataloaders
    # ============================================
    print("\n🔧 Creating datasets...")
    feature_extractor = AudioFeatureExtractor()
    
    train_dataset = WakeWordDataset(train_files, train_labels, feature_extractor)
    val_dataset = WakeWordDataset(val_files, val_labels, feature_extractor)
    test_dataset = WakeWordDataset(test_files, test_labels, feature_extractor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # ============================================
    # STEP 3: Initialize model, loss, optimizer
    # ============================================
    print("\n🧠 Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    model = create_wake_word_model(dropout_rate=0.3)
    model = model.to(device)
    
    criterion = nn.BCELoss()  # Binary Cross-Entropy for binary classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # ============================================
    # STEP 4: Training loop
    # ============================================
    print(f"\n🏋️ Training for {epochs} epochs...")
    print("="*60)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # ===== TRAINING =====
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # ===== VALIDATION =====
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.2f}%")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            output_path = Path(output_dir) / "custom_wake_word.pth"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            print(f"   ✅ Best model saved (val_loss: {avg_val_loss:.4f})")
    
    # ============================================
    # STEP 5: Final evaluation on test set
    # ============================================
    print("\n" + "="*60)
    print("🧪 FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            predictions = (outputs > 0.5).float()
            
            test_correct += (predictions == labels).sum().item()
            test_total += labels.size(0)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_accuracy = 100 * test_correct / test_total
    
    print(f"\n✅ Test Accuracy: {test_accuracy:.2f}%")
    print(f"   Correct: {test_correct}/{test_total}")
    
    # Calculate precision, recall
    all_predictions = np.array(all_predictions).flatten()
    all_labels = np.array(all_labels).flatten()
    
    true_positives = np.sum((all_predictions == 1) & (all_labels == 1))
    false_positives = np.sum((all_predictions == 1) & (all_labels == 0))
    false_negatives = np.sum((all_predictions == 0) & (all_labels == 1))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    print(f"   Precision: {precision:.2%} (of detected wake words, how many were correct)")
    print(f"   Recall: {recall:.2%} (of all wake words, how many did we detect)")
    
    # ============================================
    # STEP 6: Plot training curves
    # ============================================
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = Path(output_dir) / "training_curves.png"
    plt.savefig(plot_path)
    print(f"\n📊 Training curves saved: {plot_path}")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"📁 Model saved to: {output_path}")
    print(f"🎯 Final test accuracy: {test_accuracy:.2f}%")


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    train_model(
        data_dir="wake_word_training/data",
        output_dir="../../models",
        epochs=50,
        batch_size=16,
        learning_rate=0.001
    )
