import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler

class CyclicalLogAnnealing(_LRScheduler):
    def __init__(
        self, 
        optimizer, 
        min_lr=0.001, 
        max_lr=0.1, 
        total_epochs=100, 
        restart_interval=10, 
        restart_interval_multiplier=2.0,
        warmup_epochs=5,
        warmup_start_lr=0.0001
    ):
        """
        Implements Cyclical Log Annealing learning rate scheduler
        
        Args:
            optimizer (torch.optim.Optimizer): Wrapped optimizer
            min_lr (float): Minimum learning rate
            max_lr (float): Maximum learning rate
            total_epochs (int): Total number of training epochs
            restart_interval (int): Number of epochs between restarts
            restart_interval_multiplier (float): Multiplier for restart intervals
            warmup_epochs (int): Number of epochs for learning rate warmup
            warmup_start_lr (float): Initial learning rate for warmup
        """
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_epochs = total_epochs
        self.restart_interval = restart_interval
        self.restart_interval_multiplier = restart_interval_multiplier
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        
        super().__init__(optimizer)

    def get_lr(self):
        # Warmup phase
        if self.last_epoch < self.warmup_epochs:
            return [self._warmup_lr(self.last_epoch) for _ in self.base_lrs]
        
        # Calculate current restart interval
        current_restart_interval = self.restart_interval * (
            self.restart_interval_multiplier ** (self.last_epoch // self.restart_interval)
        )
        
        # Calculate progress within current restart cycle
        progress = (self.last_epoch % current_restart_interval) / current_restart_interval
        
        # Log annealing calculation (based on Equation 4 in the paper)
        lr_scale = 1 + np.log(max(1, current_restart_interval * progress)) / np.log(current_restart_interval)
        
        return [
            self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 - abs(lr_scale - 1))
            for _ in self.base_lrs
        ]

    def _warmup_lr(self, epoch):
        """Linear warmup of learning rate"""
        warmup_factor = (epoch + 1) / self.warmup_epochs
        return [
            self.warmup_start_lr + warmup_factor * (base_lr - self.warmup_start_lr)
            for base_lr in self.base_lrs
        ]