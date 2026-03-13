import sys
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
import torch
from model.tinygpt import TinyGPT


def get_optimizer(model, lr=1e-3, weight_decay=0.01):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_optimizer_raw(params, lr=1e-3, weight_decay=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    # Initialize states
    m = [torch.zeros_like(p, device=p.device) for p in params]
    v = [torch.zeros_like(p, device=p.device) for p in params]
    step = 0

    def step_optimizer():
        nonlocal step
        step += 1
        for i, p in enumerate(params):
            if p.grad is None:
                continue
            g = p.grad
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * (g ** 2)

            # Bias correction
            m_hat = m[i] / (1 - beta1 ** step)
            v_hat = v[i] / (1 - beta2 ** step)

            p.data = p.data - lr * (m_hat / (torch.sqrt(v_hat) + eps) + weight_decay * p.data)

            p.grad.zero_()

    return step_optimizer

