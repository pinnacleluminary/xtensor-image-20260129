"""
Adaptive hyperparameter optimization for Image AutoML fine-tuning.
Automatically adjusts training parameters based on dataset characteristics and training progress.
"""

import os
from typing import Dict, Any, Tuple
import math


class AdaptiveConfigOptimizer:
    """Optimizes training configuration based on dataset characteristics and training type."""
    
    def __init__(self):
        self.style_config = {
            "base_epochs": 25,
            "base_lr_unet": 3e-5,
            "base_lr_text": 3e-6,
            "base_batch_size": 4,
            "base_repeats": 10,
            "min_snr_gamma": 7,
            "noise_offset": 0.0411,
        }
        
        self.person_config = {
            "base_epochs": 20,
            "base_lr_unet": 4e-5,
            "base_lr_text": 4e-6,
            "base_batch_size": 2,
            "base_repeats": 8,
            "min_snr_gamma": 5,
            "noise_offset": 0.0357,
        }
    
    def count_images(self, train_data_dir: str) -> int:
        """Count the number of training images."""
        if not os.path.exists(train_data_dir):
            return 0
        
        count = 0
        for root, dirs, files in os.walk(train_data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    count += 1
        return count
    
    def determine_training_type(self, train_data_dir: str, is_style: bool) -> str:
        """Determine if training is for style or person/concept based on dataset."""
        image_count = self.count_images(train_data_dir)
        
        if is_style:
            return "style"
        
        if image_count < 15:
            return "style"
        elif image_count >= 15 and image_count < 30:
            return "person_medium"
        else:
            return "person_large"
    
    def calculate_adaptive_epochs(self, base_epochs: int, image_count: int, training_type: str) -> int:
        """Calculate optimal epochs based on dataset size and training type."""
        if training_type == "style":
            if image_count < 10:
                return min(base_epochs + 5, 30)
            elif image_count < 20:
                return base_epochs
            else:
                return max(base_epochs - 5, 15)
        else:
            if image_count < 20:
                return max(base_epochs - 5, 15)
            elif image_count < 30:
                return base_epochs
            else:
                return max(base_epochs - 3, 18)
    
    def calculate_adaptive_lr(self, base_lr: float, image_count: int, training_type: str) -> float:
        """Calculate adaptive learning rate based on dataset size."""
        if training_type == "style":
            if image_count < 10:
                return base_lr * 0.8
            elif image_count > 30:
                return base_lr * 1.2
            return base_lr
        else:
            if image_count < 15:
                return base_lr * 0.9
            elif image_count > 25:
                return base_lr * 1.1
            return base_lr
    
    def calculate_adaptive_batch_size(self, base_batch_size: int, image_count: int, available_memory_gb: float = 24.0) -> int:
        """Calculate adaptive batch size based on dataset and available memory."""
        if image_count < 10:
            return max(base_batch_size - 1, 1)
        elif image_count > 50 and available_memory_gb >= 40:
            return min(base_batch_size + 2, 8)
        elif image_count > 30 and available_memory_gb >= 24:
            return min(base_batch_size + 1, 6)
        return base_batch_size
    
    def calculate_adaptive_repeats(self, base_repeats: int, image_count: int, training_type: str) -> int:
        """Calculate adaptive repeats based on dataset size."""
        if training_type == "style":
            if image_count < 10:
                return min(base_repeats + 2, 15)
            elif image_count > 30:
                return max(base_repeats - 2, 8)
            return base_repeats
        else:
            if image_count < 15:
                return min(base_repeats + 1, 10)
            elif image_count > 30:
                return max(base_repeats - 1, 6)
            return base_repeats
    
    def calculate_gradient_accumulation(self, batch_size: int, target_effective_batch: int = 8) -> int:
        """Calculate gradient accumulation steps to achieve target effective batch size."""
        if batch_size >= target_effective_batch:
            return 1
        return max(1, math.ceil(target_effective_batch / batch_size))
    
    def optimize_config(
        self, 
        config: Dict[str, Any], 
        train_data_dir: str, 
        is_style: bool,
        model_type: str = "sdxl"
    ) -> Dict[str, Any]:
        """Optimize configuration based on dataset characteristics."""
        training_type = self.determine_training_type(train_data_dir, is_style)
        image_count = self.count_images(train_data_dir)
        
        base_config = self.style_config if training_type == "style" else self.person_config
        
        optimized_config = config.copy()
        
        base_epochs = base_config["base_epochs"]
        optimized_epochs = self.calculate_adaptive_epochs(base_epochs, image_count, training_type)
        optimized_config["max_train_epochs"] = optimized_epochs
        
        base_lr_unet = base_config["base_lr_unet"]
        base_lr_text = base_config["base_lr_text"]
        
        optimized_lr_unet = self.calculate_adaptive_lr(base_lr_unet, image_count, training_type)
        optimized_lr_text = self.calculate_adaptive_lr(base_lr_text, image_count, training_type)
        
        optimized_config["unet_lr"] = optimized_lr_unet
        optimized_config["text_encoder_lr"] = optimized_lr_text
        
        base_batch_size = base_config["base_batch_size"]
        optimized_batch_size = self.calculate_adaptive_batch_size(base_batch_size, image_count)
        optimized_config["train_batch_size"] = optimized_batch_size
        
        gradient_accumulation = self.calculate_gradient_accumulation(optimized_batch_size)
        optimized_config["gradient_accumulation_steps"] = gradient_accumulation
        
        base_repeats = base_config["base_repeats"]
        optimized_repeats = self.calculate_adaptive_repeats(base_repeats, image_count, training_type)
        
        optimized_config["min_snr_gamma"] = base_config["min_snr_gamma"]
        optimized_config["noise_offset"] = base_config["noise_offset"]
        
        if training_type != "style":
            if model_type == "sdxl":
                optimized_config["lr_scheduler"] = "constant"
                optimized_config["optimizer_type"] = "prodigy"
                optimized_config["optimizer_args"] = [
                    "decouple=True",
                    "d_coef=1",
                    "weight_decay=0.01",
                    "use_bias_correction=True",
                    "safeguard_warmup=True"
                ]
                optimized_config["unet_lr"] = 1.0
                optimized_config["text_encoder_lr"] = 1.0
            else:
                optimized_config["lr_scheduler"] = "cosine"
        
        if image_count > 40:
            optimized_config["max_grad_norm"] = 0.5
        elif image_count < 10:
            optimized_config["max_grad_norm"] = 1.5
        
        save_every_n_epochs = max(1, optimized_epochs // 5)
        optimized_config["save_every_n_epochs"] = save_every_n_epochs
        
        if image_count < 15:
            optimized_config["lr_warmup_steps"] = max(10, optimized_epochs // 4)
        else:
            optimized_config["lr_warmup_steps"] = max(20, optimized_epochs // 5)
        
        if training_type == "style" and image_count < 12:
            optimized_config["caption_dropout_rate"] = 0.05
        elif training_type != "style":
            optimized_config["caption_dropout_rate"] = 0.1
        
        if image_count > 50:
            optimized_config["max_data_loader_n_workers"] = min(4, optimized_config.get("max_data_loader_n_workers", 2) + 1)
        
        return optimized_config, optimized_repeats
    
    def get_early_stopping_config(self, image_count: int, training_type: str) -> Dict[str, Any]:
        """Get early stopping configuration."""
        if training_type == "style":
            patience = 5 if image_count < 15 else 7
            min_delta = 0.001
        else:
            patience = 3 if image_count < 20 else 4
            min_delta = 0.002
        
        return {
            "early_stopping_patience": patience,
            "early_stopping_min_delta": min_delta,
            "early_stopping_monitor": "loss"
        }
