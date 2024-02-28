"""Implements an Adapter"""
import torch
import torch.nn as nn

from utils import init_linear_layer, get_activation

class Adapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, input_dim):
        super().__init__()
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        self.input_dim = input_dim
        # config.adapter_reduction_factor passing
        self.down_sample_size = self.input_dim // 64
        # config.adapter_non_linearity.lower()
        self.activation = get_activation("gelu")
        self.down_sampler = init_linear_layer(self.input_dim, self.down_sample_size).to(self.device)
        self.up_sampler = init_linear_layer(self.down_sample_size, self.input_dim).to(self.device)

    def forward(self, x):
        x = self.down_sampler(x)
        x = self.activation(x)
        x = self.up_sampler(x)
        return x

