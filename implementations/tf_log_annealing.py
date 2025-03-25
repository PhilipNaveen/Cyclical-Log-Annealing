import tensorflow as tf
import numpy as np

class TensorFlowLogAnnealing(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, 
        initial_learning_rate=0.1,
        min_learning_rate=0.001,
        max_learning_rate=0.1, 
        total_epochs=100, 
        restart_interval=10,
        restart_interval_multiplier=2.0,
        warmup_epochs=5,
        warmup_start_lr=0.0001
    ):
        """
        Implements Cyclical Log Annealing learning rate scheduler for TensorFlow
        
        Args:
            initial_learning_rate (float): Initial learning rate
            min_learning_rate (float): Minimum learning rate
            max_learning_rate (float): Maximum learning rate
            total_epochs (int): Total number of training epochs
            restart_interval (int): Number of epochs between restarts
            restart_interval_multiplier (float): Multiplier for restart intervals
            warmup_epochs (int): Number of epochs for learning rate warmup
            warmup_start_lr (float): Initial learning rate for warmup
        """
        super().__init__()
        
        self.initial_learning_rate = initial_learning_rate
        self.min_lr = min_learning_rate
        self.max_lr = max_learning_rate
        self.total_epochs = total_epochs
        self.restart_interval = restart_interval
        self.restart_interval_multiplier = restart_interval_multiplier
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr

    def __call__(self, step):
        # Convert step to epoch
        epoch = tf.cast(step / (step + 1), tf.float32) * self.total_epochs
        
        # Warmup phase
        warmup_condition = tf.less(epoch, tf.cast(self.warmup_epochs, tf.float32))
        warmup_lr = self._warmup_lr(epoch)
        
        # Log annealing calculation
        def log_annealing_lr():
            # Calculate current restart interval
            current_restart_interval = self.restart_interval * (
                self.restart_interval_multiplier ** tf.cast(tf.floor(epoch / self.restart_interval), tf.float32)
            )
            
            # Calculate progress within current restart cycle
            progress = tf.math.mod(epoch, current_restart_interval) / current_restart_interval
            
            # Log annealing calculation
            lr_scale = 1 + tf.math.log(tf.maximum(1.0, current_restart_interval * progress)) / tf.math.log(current_restart_interval)
            
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 - tf.abs(lr_scale - 1))
        
        # Return warmup LR during warmup, otherwise use log annealing
        return tf.cond(warmup_condition, lambda: warmup_lr, log_annealing_lr)

    def _warmup_lr(self, epoch):
        """Linear warmup of learning rate"""
        warmup_factor = (epoch + 1) / self.warmup_epochs
        return self.warmup_start_lr + warmup_factor * (self.initial_learning_rate - self.warmup_start_lr)

    def get_config(self):
        """
        Allows the learning rate scheduler to be serialized
        """
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "min_learning_rate": self.min_lr,
            "max_learning_rate": self.max_lr,
            "total_epochs": self.total_epochs,
            "restart_interval": self.restart_interval,
            "restart_interval_multiplier": self.restart_interval_multiplier,
            "warmup_epochs": self.warmup_epochs,
            "warmup_start_lr": self.warmup_start_lr
        }