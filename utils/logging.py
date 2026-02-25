import os
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir="outputs/logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
    
    def log_metrics(self, metrics: dict, step: int):
        """
        metrics: dict of name -> value
        step: training step or epoch
        """
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        
    def close(self):
        self.writer.close()
        