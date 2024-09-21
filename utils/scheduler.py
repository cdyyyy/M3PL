import numpy as np

def assign_learning_rate(param_group, lr):
    """Assign learning rate to param group."""
    param_group["lr"] = lr

def _warmup_lr(base_lr, warmup_steps, step):
    """Warmup learning rate."""
    return base_lr * step / warmup_steps

def cosine_scheduler(optimizer, base_lrs, warmup_steps, max_steps, min_lr=0.0):
    """Cosine learning rate scheduler."""
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs] * len(optimizer.param_groups)
    assert len(optimizer.param_groups) == len(base_lrs)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_steps:
                lr = _warmup_lr(base_lr, warmup_steps, step)
            else:
                lr = min_lr + 0.5 * base_lr * (
                    1 + np.cos((step - warmup_steps) / (max_steps - warmup_steps) * np.pi)
                )
            assign_learning_rate(param_group, lr)

    return _lr_adjuster