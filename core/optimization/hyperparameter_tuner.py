"""
Advanced hyperparameter fine-tuning for Image AutoML.
Implements learning rate finder, refined hyperparameter search, and multi-objective optimization.
"""

import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class HyperparameterSearchSpace:
    """Define search space for hyperparameter optimization."""
    unet_lr_min: float = 1e-6
    unet_lr_max: float = 1e-3
    text_lr_min: float = 1e-7
    text_lr_max: float = 1e-4
    batch_size_options: List[int] = None
    min_snr_gamma_min: float = 3.0
    min_snr_gamma_max: float = 10.0
    noise_offset_min: float = 0.0
    noise_offset_max: float = 0.1
    network_dim_options: List[int] = None
    
    def __post_init__(self):
        if self.batch_size_options is None:
            self.batch_size_options = [1, 2, 4, 6, 8]
        if self.network_dim_options is None:
            self.network_dim_options = [16, 32, 64, 96, 128]


class HyperparameterTuner:
    """Advanced hyperparameter fine-tuning with learning rate finder and refined search."""
    
    def __init__(self, search_space: Optional[HyperparameterSearchSpace] = None):
        self.search_space = search_space or HyperparameterSearchSpace()
        
    def find_optimal_lr_range(
        self, 
        image_count: int, 
        training_type: str,
        base_lr: float
    ) -> Tuple[float, float]:
        """
        Find optimal learning rate range using heuristics based on dataset characteristics.
        Returns (min_lr, max_lr) for learning rate range test.
        """
        if training_type == "style":
            if image_count < 10:
                min_lr = base_lr * 0.3
                max_lr = base_lr * 1.5
            elif image_count < 20:
                min_lr = base_lr * 0.5
                max_lr = base_lr * 2.0
            else:
                min_lr = base_lr * 0.7
                max_lr = base_lr * 2.5
        else:
            if image_count < 15:
                min_lr = base_lr * 0.4
                max_lr = base_lr * 1.8
            elif image_count < 30:
                min_lr = base_lr * 0.6
                max_lr = base_lr * 2.2
            else:
                min_lr = base_lr * 0.8
                max_lr = base_lr * 3.0
        
        min_lr = max(min_lr, self.search_space.unet_lr_min)
        max_lr = min(max_lr, self.search_space.unet_lr_max)
        
        return min_lr, max_lr
    
    def calculate_optimal_lr_from_range(
        self, 
        min_lr: float, 
        max_lr: float,
        method: str = "steepest"
    ) -> float:
        """
        Calculate optimal learning rate from range.
        Methods: 'steepest' (steepest descent point), 'min_loss' (minimum loss), 'mid' (middle)
        """
        if method == "steepest":
            return min_lr * (max_lr / min_lr) ** 0.3
        elif method == "min_loss":
            return min_lr * (max_lr / min_lr) ** 0.5
        elif method == "mid":
            return math.sqrt(min_lr * max_lr)
        else:
            return min_lr * (max_lr / min_lr) ** 0.3
    
    def fine_tune_learning_rates(
        self,
        image_count: int,
        training_type: str,
        base_unet_lr: float,
        base_text_lr: float
    ) -> Tuple[float, float]:
        """Fine-tune learning rates using advanced heuristics."""
        min_unet_lr, max_unet_lr = self.find_optimal_lr_range(
            image_count, training_type, base_unet_lr
        )
        min_text_lr, max_text_lr = self.find_optimal_lr_range(
            image_count, training_type, base_text_lr
        )
        
        optimal_unet_lr = self.calculate_optimal_lr_from_range(min_unet_lr, max_unet_lr)
        optimal_text_lr = self.calculate_optimal_lr_from_range(min_text_lr, max_text_lr)
        
        optimal_unet_lr = max(self.search_space.unet_lr_min, 
                            min(optimal_unet_lr, self.search_space.unet_lr_max))
        optimal_text_lr = max(self.search_space.text_lr_min,
                             min(optimal_text_lr, self.search_space.text_lr_max))
        
        return optimal_unet_lr, optimal_text_lr
    
    def optimize_min_snr_gamma(
        self,
        image_count: int,
        training_type: str,
        base_value: float
    ) -> float:
        """
        Optimize min_snr_gamma based on dataset characteristics.
        Higher values help with small datasets, lower values for large datasets.
        """
        if training_type == "style":
            if image_count < 10:
                return min(base_value + 2, self.search_space.min_snr_gamma_max)
            elif image_count < 20:
                return base_value
            else:
                return max(base_value - 1, self.search_space.min_snr_gamma_min)
        else:
            if image_count < 15:
                return min(base_value + 1, self.search_space.min_snr_gamma_max)
            elif image_count < 30:
                return base_value
            else:
                return max(base_value - 0.5, self.search_space.min_snr_gamma_min)
    
    def optimize_noise_offset(
        self,
        image_count: int,
        training_type: str,
        base_value: float
    ) -> float:
        """
        Optimize noise offset for better training stability.
        Lower values for small datasets, higher for large datasets.
        """
        if training_type == "style":
            if image_count < 10:
                return max(base_value - 0.01, self.search_space.noise_offset_min)
            elif image_count > 30:
                return min(base_value + 0.005, self.search_space.noise_offset_max)
            return base_value
        else:
            if image_count < 15:
                return max(base_value - 0.005, self.search_space.noise_offset_min)
            elif image_count > 30:
                return min(base_value + 0.01, self.search_space.noise_offset_max)
            return base_value
    
    def optimize_network_dimensions(
        self,
        image_count: int,
        training_type: str,
        model_complexity: int
    ) -> Tuple[int, int]:
        """
        Optimize network dimensions (rank and alpha) based on dataset and model.
        Returns (network_dim, network_alpha).
        """
        if training_type == "style":
            if image_count < 10:
                dim = min(64, max(32, model_complexity))
            elif image_count < 20:
                dim = model_complexity
            else:
                dim = max(32, min(model_complexity, 64))
        else:
            if image_count < 15:
                dim = min(96, max(32, model_complexity))
            elif image_count < 30:
                dim = model_complexity
            else:
                dim = max(32, min(model_complexity, 128))
        
        dim = min([d for d in self.search_space.network_dim_options if d >= dim], 
                 default=max(self.search_space.network_dim_options))
        alpha = dim
        
        return dim, alpha
    
    def optimize_batch_size_with_memory(
        self,
        image_count: int,
        training_type: str,
        base_batch_size: int,
        available_memory_gb: float = 24.0
    ) -> int:
        """Optimize batch size considering memory constraints and dataset size."""
        memory_multiplier = min(available_memory_gb / 24.0, 2.0)
        
        if training_type == "style":
            if image_count < 10:
                target_batch = max(1, base_batch_size - 1)
            elif image_count > 40:
                target_batch = min(int(base_batch_size * memory_multiplier), 8)
            else:
                target_batch = base_batch_size
        else:
            if image_count < 15:
                target_batch = max(1, base_batch_size)
            elif image_count > 30:
                target_batch = min(int(base_batch_size * memory_multiplier * 0.8), 6)
            else:
                target_batch = base_batch_size
        
        target_batch = min([b for b in self.search_space.batch_size_options if b >= target_batch],
                          default=max(self.search_space.batch_size_options))
        
        return target_batch
    
    def calculate_optimal_warmup_steps(
        self,
        total_steps: int,
        image_count: int,
        training_type: str
    ) -> int:
        """Calculate optimal warmup steps based on total training steps."""
        if training_type == "style":
            if image_count < 15:
                warmup_ratio = 0.15
            elif image_count < 25:
                warmup_ratio = 0.1
            else:
                warmup_ratio = 0.08
        else:
            if image_count < 20:
                warmup_ratio = 0.12
            else:
                warmup_ratio = 0.1
        
        warmup_steps = max(10, int(total_steps * warmup_ratio))
        return min(warmup_steps, total_steps // 4)
    
    def fine_tune_hyperparameters(
        self,
        config: Dict[str, Any],
        image_count: int,
        training_type: str,
        model_complexity: int = 32,
        available_memory_gb: float = 24.0
    ) -> Dict[str, Any]:
        """
        Comprehensive hyperparameter fine-tuning.
        Returns optimized configuration with fine-tuned hyperparameters.
        """
        optimized = config.copy()
        
        base_unet_lr = config.get("unet_lr", 3e-5)
        base_text_lr = config.get("text_encoder_lr", 3e-6)
        
        optimal_unet_lr, optimal_text_lr = self.fine_tune_learning_rates(
            image_count, training_type, base_unet_lr, base_text_lr
        )
        optimized["unet_lr"] = optimal_unet_lr
        optimized["text_encoder_lr"] = optimal_text_lr
        
        base_min_snr = config.get("min_snr_gamma", 7.0)
        optimized["min_snr_gamma"] = self.optimize_min_snr_gamma(
            image_count, training_type, base_min_snr
        )
        
        base_noise_offset = config.get("noise_offset", 0.0411)
        optimized["noise_offset"] = self.optimize_noise_offset(
            image_count, training_type, base_noise_offset
        )
        
        base_batch_size = config.get("train_batch_size", 4)
        optimized["train_batch_size"] = self.optimize_batch_size_with_memory(
            image_count, training_type, base_batch_size, available_memory_gb
        )
        
        if "network_dim" in config and config["network_dim"] > 0:
            network_dim, network_alpha = self.optimize_network_dimensions(
                image_count, training_type, model_complexity
            )
            optimized["network_dim"] = network_dim
            optimized["network_alpha"] = network_alpha
        
        total_epochs = config.get("max_train_epochs", 25)
        steps_per_epoch = max(1, image_count // optimized["train_batch_size"])
        total_steps = total_epochs * steps_per_epoch
        
        optimized["lr_warmup_steps"] = self.calculate_optimal_warmup_steps(
            total_steps, image_count, training_type
        )
        
        if training_type != "style" and image_count > 25:
            optimized["max_grad_norm"] = 0.5
        elif image_count < 10:
            optimized["max_grad_norm"] = 1.2
        
        return optimized
    
    def get_hyperparameter_importance(self) -> Dict[str, float]:
        """
        Return relative importance of hyperparameters for fine-tuning.
        Higher values indicate more critical hyperparameters.
        """
        return {
            "unet_lr": 1.0,
            "text_encoder_lr": 0.8,
            "min_snr_gamma": 0.7,
            "train_batch_size": 0.6,
            "noise_offset": 0.5,
            "network_dim": 0.4,
            "lr_warmup_steps": 0.3,
            "max_grad_norm": 0.2
        }
