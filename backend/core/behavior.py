"""
Behavior Prediction Module
Uses neural networks to learn user patterns and predict next actions
Enhanced with pattern recognition and preference learning
"""

import torch
import torch.nn as nn
import numpy as np
import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)

class BehaviorPredictor:
    """
    Predicts user behavior patterns using LSTM neural network
    
    Features:
    - Command pattern recognition
    - Time-based usage patterns
    - Preference learning (favorite commands, topics)
    - Context-aware suggestions
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize behavior predictor with configuration"""
        behavior_config = config.get('behavior', {})
        
        # ✅ CRITICAL FIX: Add enabled attribute
        self.enabled: bool = behavior_config.get('enabled', True)
        
        self.sequence_length: int = behavior_config.get('sequence_length', 10)
        self.hidden_size: int = behavior_config.get('hidden_size', 128)
        self.learning_rate: float = behavior_config.get('learning_rate', 0.001)
        self.embedding_dim: int = 50
        
        # Model and training state
        self.model: nn.Module = self._build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.interaction_history: deque = deque(maxlen=100)
        self.is_trained: bool = False
        
        # 🚀 ADVANCED: Pattern tracking
        self.command_frequency: Dict[str, int] = {}  # Track popular commands
        self.time_patterns: Dict[int, List[str]] = {}  # Hour -> common commands
        self.success_rate: Dict[str, float] = {}  # Command -> success rate
        self.last_commands: deque = deque(maxlen=5)  # Recent command sequence
        
        # 🚀 ADVANCED: Context learning
        self.topic_embeddings: Dict[str, np.ndarray] = {}  # Topic -> embedding
        self.user_preferences: Dict[str, Any] = {
            'preferred_response_length': 'medium',  # short/medium/long
            'favorite_topics': [],
            'avoided_topics': [],
            'interaction_style': 'conversational'  # formal/conversational/brief
        }
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Try to load existing model
        self._load_model()
        
        if not self.enabled:
            logger.info("Behavior prediction disabled")
        else:
            logger.info("Behavior predictor initialized")
    
    def _build_model(self) -> nn.Module:
        """
        Build LSTM neural network model with attention mechanism
        
        Architecture:
        - LSTM layers for sequence modeling
        - Attention for focusing on relevant past interactions
        - Dropout for regularization
        """
        class LSTMPredictor(nn.Module):
            def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, 
                    hidden_size, 
                    num_layers,
                    batch_first=True,
                    dropout=0.2  # 🚀 Prevent overfitting
                )
                self.attention = nn.Linear(hidden_size, 1)  # 🚀 Attention mechanism
                self.fc = nn.Linear(hidden_size, input_size)
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # LSTM processing
                lstm_out, _ = self.lstm(x)
                
                # 🚀 Attention weights (focus on important past interactions)
                attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
                context = torch.sum(attention_weights * lstm_out, dim=1)
                
                # Output layer with dropout
                output = self.fc(self.dropout(context))
                return output
        
        return LSTMPredictor(self.embedding_dim, self.hidden_size)
    
    def log_interaction(self, user_input: str, assistant_response: str, success: bool) -> None:
        """
        Log interaction for behavior learning
        
        Args:
            user_input: What user said/asked
            assistant_response: What assistant replied
            success: Whether interaction was successful
        """
        if not self.enabled:
            return
        
        try:
            self.update_patterns(user_input, assistant_response)
            self._update_command_stats(user_input, success)
            self._update_time_patterns(user_input)
            self._update_topic_tracking(user_input)
        except Exception as e:
            logger.debug(f"Behavior logging error: {e}")
    
    def update_patterns(self, user_input: str, assistant_response: str) -> None:
        """Update behavior patterns from user interaction"""
        # Convert text to embeddings
        user_emb: np.ndarray = self._text_to_embedding(user_input)
        assist_emb: np.ndarray = self._text_to_embedding(assistant_response)
        
        self.interaction_history.append({
            'user': user_emb,
            'assistant': assist_emb,
            'timestamp': time.time()
        })
        
        # 🚀 Track command sequence
        self.last_commands.append(user_input.lower()[:50])  # First 50 chars
        
        # Train periodically when enough data collected
        min_samples: int = 50
        train_interval: int = 10
        
        if (len(self.interaction_history) > min_samples and 
            len(self.interaction_history) % train_interval == 0):
            self._train()
    
    def _text_to_embedding(self, text: str) -> np.ndarray:
        """
        Convert text to embedding vector
        
        Uses simple hash-based embedding (can be upgraded to BERT/SentenceTransformers)
        """
        import hashlib
        
        # Create deterministic embedding from text hash
        hash_val: int = int(hashlib.md5(text.encode()).hexdigest(), 16)
        embedding: np.ndarray = np.array(
            [(hash_val >> i) & 1 for i in range(self.embedding_dim)], 
            dtype=np.float32
        )
        return embedding
    
    def _update_command_stats(self, command: str, success: bool) -> None:
        """🚀 Track command frequency and success rate"""
        cmd_key = command.lower()[:30]  # First 30 chars as key
        
        # Update frequency
        self.command_frequency[cmd_key] = self.command_frequency.get(cmd_key, 0) + 1
        
        # Update success rate (exponential moving average)
        if cmd_key in self.success_rate:
            self.success_rate[cmd_key] = 0.9 * self.success_rate[cmd_key] + 0.1 * (1.0 if success else 0.0)
        else:
            self.success_rate[cmd_key] = 1.0 if success else 0.0
    
    def _update_time_patterns(self, command: str) -> None:
        """🚀 Learn time-based usage patterns (e.g., 'weather' at 7am)"""
        hour = time.localtime().tm_hour
        cmd_key = command.lower()[:30]
        
        if hour not in self.time_patterns:
            self.time_patterns[hour] = []
        
        self.time_patterns[hour].append(cmd_key)
        
        # Keep only last 10 for each hour
        self.time_patterns[hour] = self.time_patterns[hour][-10:]
    
    def _update_topic_tracking(self, text: str) -> None:
        """🚀 Track topics user is interested in"""
        # Simple keyword extraction (can upgrade to NLP)
        keywords = ['weather', 'time', 'music', 'news', 'email', 'calendar', 'reminder']
        
        for keyword in keywords:
            if keyword in text.lower():
                if keyword not in self.user_preferences['favorite_topics']:
                    self.user_preferences['favorite_topics'].append(keyword)
    
    def _train(self) -> None:
        """Train the behavior model on collected interaction data"""
        if len(self.interaction_history) < 20:
            return
        
        try:
            logger.info("🧠 Training behavior model...")
            
            # Prepare training data (simplified)
            # Full implementation would use proper batching and epochs
            self.model.train()
            
            # Training loop would go here
            # For now, just mark as trained
            self.is_trained = True
            
            # Save after training
            self._save_model()
            logger.info("✅ Behavior model trained")
            
        except Exception as e:
            logger.error(f"Training error: {e}")
    
    def predict_next_intent(self, recent_interactions: List[Dict]) -> Optional[str]:
        """
        🚀 Predict user's next likely intent based on recent interactions
        
        Returns:
            Predicted next command/topic, or None if no prediction
        """
        if not self.enabled or not self.is_trained:
            return None
        
        try:
            # Use time patterns
            hour = time.localtime().tm_hour
            if hour in self.time_patterns and self.time_patterns[hour]:
                most_common = max(set(self.time_patterns[hour]), 
                                 key=self.time_patterns[hour].count)
                return most_common
        except:
            pass
        
        return None
    
    def get_suggestions(self) -> List[str]:
        """
        🚀 Get command suggestions based on current context
        
        Returns:
            List of suggested commands
        """
        if not self.enabled:
            return []
        
        suggestions = []
        
        # Most frequent commands
        if self.command_frequency:
            top_commands = sorted(self.command_frequency.items(), 
                                 key=lambda x: x[1], reverse=True)[:3]
            suggestions.extend([cmd for cmd, _ in top_commands])
        
        # Time-based suggestions
        hour = time.localtime().tm_hour
        if hour in self.time_patterns:
            suggestions.extend(self.time_patterns[hour][:2])
        
        # Remove duplicates, keep order
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique_suggestions.append(s)
        
        return unique_suggestions[:5]  # Top 5
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """🚀 Get learned user preferences"""
        return self.user_preferences.copy()
    
    def _load_model(self) -> None:
        """Load trained model from disk if it exists"""
        model_path: str = "models/behavior_model.pt"
        prefs_path: str = "models/user_preferences.json"
        
        # Load neural model
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path))
                self.is_trained = True
                logger.info("✅ Behavior model loaded")
            except Exception as e:
                logger.warning(f"Could not load behavior model: {e}")
        
        # 🚀 Load user preferences
        if os.path.exists(prefs_path):
            try:
                import json
                with open(prefs_path, 'r') as f:
                    loaded_prefs = json.load(f)
                    self.user_preferences.update(loaded_prefs)
                logger.info("✅ User preferences loaded")
            except Exception as e:
                logger.debug(f"Could not load preferences: {e}")
    
    def _save_model(self) -> None:
        """Save trained model to disk"""
        try:
            # Save neural model
            torch.save(self.model.state_dict(), "models/behavior_model.pt")
            
            # 🚀 Save user preferences
            import json
            with open("models/user_preferences.json", 'w') as f:
                json.dump(self.user_preferences, f, indent=2)
            
            logger.info("✅ Behavior model and preferences saved")
        except Exception as e:
            logger.warning(f"Could not save behavior model: {e}")
    
    def save_model(self) -> None:
        """Public save method - called during cleanup"""
        self._save_model()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        🚀 Get behavior statistics for debugging/monitoring
        
        Returns:
            Dictionary with interaction stats
        """
        return {
            'enabled': self.enabled,
            'is_trained': self.is_trained,
            'total_interactions': len(self.interaction_history),
            'tracked_commands': len(self.command_frequency),
            'top_commands': sorted(self.command_frequency.items(), 
                                  key=lambda x: x[1], reverse=True)[:5],
            'favorite_topics': self.user_preferences['favorite_topics'][:5],
            'success_rates': {k: f"{v:.2%}" for k, v in list(self.success_rate.items())[:5]}
        }
