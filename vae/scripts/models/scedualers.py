import torch
from torch.optim.lr_scheduler import _LRScheduler

class WarmupExponentialDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, decay_rate, decay_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        super(WarmupExponentialDecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # Exponential decay
            decay_factor = self.decay_rate ** ((self.last_epoch - self.warmup_steps) / self.decay_steps)
            return [base_lr * decay_factor for base_lr in self.base_lrs]