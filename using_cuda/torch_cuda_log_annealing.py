import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

class PyTorchLogAnnealing(_LRScheduler):
    def __init__(
        self, 
        optimizer, 
        min_lr=0.001, 
        max_lr=0.1, 
        total_epochs=100, 
        restart_interval=10, 
        restart_interval_multiplier=2.0,
        warmup_epochs=5,
        warmup_start_lr=0.0001,
        device=None
    ):
        """
        Implements Cyclical Log Annealing learning rate scheduler with CUDA support
        
        Args:
            optimizer (torch.optim.Optimizer): Wrapped optimizer
            min_lr (float): Minimum learning rate
            max_lr (float): Maximum learning rate
            total_epochs (int): Total number of training epochs
            restart_interval (int): Number of epochs between restarts
            restart_interval_multiplier (float): Multiplier for restart intervals
            warmup_epochs (int): Number of epochs for learning rate warmup
            warmup_start_lr (float): Initial learning rate for warmup
            device (torch.device, optional): Device to use for computations
        """
        # Automatically detect device if not specified
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.min_lr = torch.tensor(min_lr, device=device)
        self.max_lr = torch.tensor(max_lr, device=device)
        self.total_epochs = total_epochs
        self.restart_interval = restart_interval
        self.restart_interval_multiplier = restart_interval_multiplier
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = torch.tensor(warmup_start_lr, device=device)
        self.device = device
        
        super().__init__(optimizer)

    def get_lr(self):
        # Warmup phase
        if self.last_epoch < self.warmup_epochs:
            return self._warmup_lr()
        
        # Calculate current restart interval
        current_restart_interval = self.restart_interval * (
            self.restart_interval_multiplier ** (self.last_epoch // self.restart_interval)
        )
        
        # Calculate progress within current restart cycle
        progress = (self.last_epoch % current_restart_interval) / current_restart_interval
        
        # Log annealing calculation with torch operations
        current_restart_interval_tensor = torch.tensor(current_restart_interval, device=self.device)
        progress_tensor = torch.tensor(progress, device=self.device)
        
        lr_scale = 1 + torch.log(torch.max(torch.tensor(1.0, device=self.device), 
                                            current_restart_interval_tensor * progress_tensor)) / torch.log(current_restart_interval_tensor)
        
        return [
            self.min_lr.item() + 0.5 * (self.max_lr.item() - self.min_lr.item()) * (1 - torch.abs(lr_scale - 1).item())
            for _ in self.base_lrs
        ]

    def _warmup_lr(self):
        """Linear warmup of learning rate"""
        warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
        return [
            self.warmup_start_lr.item() + warmup_factor * (base_lr - self.warmup_start_lr.item())
            for base_lr in self.base_lrs
        ]

    def state_dict(self):
        """
        Ensures state dictionary is device-agnostic
        """
        state = super().state_dict()
        # Convert tensor attributes to CPU for serialization
        state['min_lr'] = self.min_lr.cpu().item()
        state['max_lr'] = self.max_lr.cpu().item()
        state['warmup_start_lr'] = self.warmup_start_lr.cpu().item()
        return state

    def load_state_dict(self, state_dict):
        """
        Loads state dictionary with proper device handling
        """
        # Convert scalar values back to tensors on the original device
        state_dict['min_lr'] = torch.tensor(state_dict['min_lr'], device=self.device)
        state_dict['max_lr'] = torch.tensor(state_dict['max_lr'], device=self.device)
        state_dict['warmup_start_lr'] = torch.tensor(state_dict['warmup_start_lr'], device=self.device)
        super().load_state_dict(state_dict)