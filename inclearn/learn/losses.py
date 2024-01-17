import torch
from torch.nn import functional as F

def energy_criterion(inlier_logits, outlier_logits, m_in, m_out):
    Ec_out = -torch.logsumexp(outlier_logits, dim=1)
    loss_out = torch.pow(F.relu(m_out - Ec_out), 2).mean()
    Ec_in = -torch.logsumexp(inlier_logits, dim=1)
    
    if torch.isnan(Ec_in.mean()):
        loss_in = 0.0
    else:
        loss_in = torch.pow(F.relu(Ec_in - m_in), 2).mean()
    loss_novelty = 0.1*(loss_in + loss_out)
    
    return loss_novelty