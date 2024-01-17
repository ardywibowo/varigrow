from typing import Optional

import torch
import torch.nn.functional as F
from inclearn.convnet.sparse_layers import SparseGate
from torch import Tensor, distributions, nn
from torch.nn import Parameter

class PruningLinear(nn.Linear):
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True
    ) -> None:
    
        super(PruningLinear, self).__init__(
            in_features, out_features, bias)
        
        self.gate = PruningLayer(out_features)

    def forward(self, input: Tensor) -> Tensor:
        out = F.linear(input, self.weight, self.bias)
        out = self.gate(out)
        
        return out

class PruningConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'
    ) -> None:
        super(PruningConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        
        self.gate = PruningLayer(out_channels)
    
    def conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            0, self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
    def forward(self, input: Tensor) -> Tensor:
        out = self.conv_forward(input, self.weight, self.bias)
        out = self.gate(out)
        
        return out

    def reg(self):
        activated_channels = self.gate.sparsity().sum()
        total_channels = self.gate.num_channel
        
        kernel_params = 1
        for k in self.kernel_size:
            kernel_params = kernel_params * k        
        
        return activated_channels * kernel_params, total_channels * kernel_params

class PruningLayer(nn.Module):
    def __init__(self, num_channel, reduction=8):
        super(PruningLayer, self).__init__()
        
        self.num_channel = num_channel
        self.gate = SparseGate()
        self.mask = Parameter(6.0 * torch.ones((num_channel)))
        
    def forward(self, x):
        x_shape = x.size()

        gates = self.gate(self.mask)
        gates = gates[None, :, None, None]
        return x * gates
    
    def sparsity(self):
        return torch.sigmoid(self.mask)
