import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, distributions, nn
from torch.nn import Parameter

class SparseGate(nn.Module):
    def __init__(self, gating_method='gumbel'):
        super(SparseGate, self).__init__()

        self.gating_method = gating_method
        self.temp = 1.0
        self.limit_bottom, self.limit_top = -0.1, 1.1
        self.threshold = 0.0
        
    def forward(self, x):
        # return nn.Sigmoid()(input)
    
        out = sparse_gate(
            x, 
            temp=self.temp, 
            limit_top=self.limit_top, 
            limit_bottom=self.limit_bottom,
            gating_method=self.gating_method,
            thres=self.threshold,
            training=self.training)

        return out
    
def l0_sample(logit, temp=2./3., limit_top=1.1, limit_bottom=-0.1):
    u = torch.rand_like(logit)
    sigm = torch.sigmoid((torch.log(u) - torch.log(1-u) + logit) / temp)
    sbar = sigm * (limit_top - limit_bottom) + limit_bottom
    return torch.clamp(sbar, 0, 1)

def gumbel_sample(logit, temp=0.1):
    u = torch.rand_like(logit)
    sigm = torch.sigmoid((torch.log(u) - torch.log(1-u) + logit) / temp)
    return sigm

def hardsigmoid(x, temp=2./3., limit_top=1.1, limit_bottom=-0.1):
    sigm = torch.sigmoid(x / temp)
    ybar = sigm * (limit_top - limit_bottom) + limit_bottom
    return torch.clamp(ybar, 0, 1)

def sparse_gate(logit, temp, limit_top, limit_bottom, gating_method, thres, training=True):
    args = []

    # if True:
    if training:
        if gating_method == 'l0':
            gate_f = l0_sample
            args = temp, limit_top, limit_bottom

        elif gating_method == 'gumbel':
            gate_f = gumbel_sample
            args = temp,

        elif gating_method == 'fixed':
            gate_f = hardsigmoid
            args = temp, limit_top, limit_bottom

        else:
            raise ValueError("Method only supports the following sparse gating methods "
                                "[l0, fixed] gates, not {}".format(
                gating_method))
    else:
        gate_f = lambda x: (x > thres).float()

    gates = gate_f(logit, *args)

    return gates
