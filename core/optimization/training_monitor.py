"""
Training monitoring and early stopping for Image AutoML fine-tuning.
Tracks training progress and implements adaptive early stopping.
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class TrainingMonitor:
    """Monitors training progress and implements early stopping."""
    
    def __init__(self, log_dir: str, patience: int = 5, min_delta: float = 0.001):
        """
        Initialize training monitor.
        
        Args:
            log_dir: Directory to save monitoring logs
            patience: Number of epochs to wait before early stopping
            min_delta: Minimum change in loss to qualify as improvement
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.loss_history: List[float] = []
        self.epoch_history: List[int] = []
        
    def log_loss(self, epoch: int, loss: float, step: Optional[int] = None) -> None:
        """Log training loss for an epoch."""
        self.loss_history.append(loss)
        self.epoch_history.append(epoch)
        
        log_entry = {
            "epoch": epoch,
            "loss": loss,
            "step": step
        }
        
        log_file = self.log_dir / "training_log.json"
        if log_file.exists():
            with open(log_file, "r") as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_entry)
        
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)
    
    def should_stop(self, current_loss: float) -> Tuple[bool, str]:
        """
        Check if training should stop early.
        
        Returns:
            Tuple of (should_stop, reason)
        """
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
            return False, "Improving"
        
        self.patience_counter += 1
        
        if self.patience_counter >= self.patience:
            return True, f"Early stopping: no improvement for {self.patience} epochs"
        
        return False, f"No improvement for {self.patience_counter}/{self.patience} epochs"
    
    def get_loss_trend(self, window: int = 5) -> str:
        """Get loss trend over recent epochs."""
        if len(self.loss_history) < window:
            return "insufficient_data"
        
        recent_losses = self.loss_history[-window:]
        if len(recent_losses) < 2:
            return "stable"
        
        first_half = sum(recent_losses[:len(recent_losses)//2]) / (len(recent_losses)//2)
        second_half = sum(recent_losses[len(recent_losses)//2:]) / (len(recent_losses) - len(recent_losses)//2)
        
        if second_half < first_half - self.min_delta:
            return "decreasing"
        elif second_half > first_half + self.min_delta:
            return "increasing"
        else:
            return "stable"
    
    def get_optimal_lr_adjustment(self) -> float:
        """Suggest learning rate adjustment based on loss trend."""
        trend = self.get_loss_trend()
        
        if trend == "increasing":
            return 0.5
        elif trend == "stable" and len(self.loss_history) > 3:
            return 0.8
        else:
            return 1.0
    
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save monitoring checkpoint."""
        checkpoint_data = {
            "best_loss": self.best_loss,
            "patience_counter": self.patience_counter,
            "loss_history": self.loss_history,
            "epoch_history": self.epoch_history
        }
        
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load monitoring checkpoint."""
        if not os.path.exists(checkpoint_path):
            return
        
        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)
        
        self.best_loss = checkpoint_data.get("best_loss", float('inf'))
        self.patience_counter = checkpoint_data.get("patience_counter", 0)
        self.loss_history = checkpoint_data.get("loss_history", [])
        self.epoch_history = checkpoint_data.get("epoch_history", [])
