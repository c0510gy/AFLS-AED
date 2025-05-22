import torch
import numpy as np
from torcheval.metrics.aggregation.auc import AUC

def compute_auc(neg: torch.Tensor, pos: torch.Tensor) -> float:
    """
    Compute ROC-AUC distinguishing neg vs. pos.
    """
    
    metric = AUC()
    metric.reset()
    metric.update(torch.tensor([0, 1, 1]), torch.tensor([0, 1, 0]))
    for t in np.linspace(min(min(neg).item(), min(pos).item()), max(max(neg).item(), max(pos).item()), 1000):
        tp = (pos >= t).float().sum().item()
        fn = (pos < t).float().sum().item()
        tn = (neg < t).float().sum().item()
        fp = (neg >= t).float().sum().item()
        if tp + fn > 0 and fp + tn > 0:
            metric.update(
                torch.tensor([fp / (fp + tn)]),
                torch.tensor([tp / (tp + fn)]),
            )
    return metric.compute().item()
