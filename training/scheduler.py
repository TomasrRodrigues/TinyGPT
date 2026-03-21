import math

class CosineLearningRateScheduler:
    def __init__(self, max_lr, total_epochs, steps_per_epoch, min_lr=0.0):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = total_epochs * steps_per_epoch

    def get_learning_rate(self, step):
        if step >= self.total_steps:
            return self.min_lr
        lr = self.min_lr + (1/2) * (self.max_lr - self.min_lr) * (1+ math.cos((step*math.pi)/self.total_steps))
        return lr

class WarmupCosineLearningRateScheduler:
    def __init__(self, max_lr, total_epochs, steps_per_epoch, warmup_epochs=5, min_lr=0.0):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = total_epochs * steps_per_epoch
        self.warmup_steps = warmup_epochs * steps_per_epoch

    def get_learning_rate(self, step):
        if step < self.warmup_steps:
            return self.max_lr * (step / self.warmup_steps)
        elif step >= self.total_steps:
            return self.min_lr
        else:
            lr = self.min_lr + (1/2) * (self.max_lr - self.min_lr) * (1 + math.cos(((step - self.warmup_steps) * math.pi) / (self.total_steps - self.warmup_steps)))
            return lr