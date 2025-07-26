import torch
from torch import nn
import torch.nn.functional as F

from typing import Tuple
import numpy as np

def get_conv_weight_and_bias(
        filter_size: Tuple[int, int],
        num_groups: int,
        input_channels: int,
        output_channels: int,
        bias: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    # assert that num_filters is divisible by num_groups
    assert input_channels % num_groups == 0, "input channels must be divisible by groups number"
    # assert that num_channels is divisible by num_groups
    assert output_channels % num_groups == 0, "output channels must be divisible by groups number"
    input_channels = input_channels // num_groups

    # initialize the weight matrix
    weight_matrix = torch.randn(output_channels, input_channels, *filter_size)
    # initialize the bias vector
    if bias:
        bias_vector = torch.ones(output_channels)
    else:
        bias_vector = None
    return weight_matrix, bias_vector


class MyConvStub:
    def __init__(
            self,
            kernel_size: Tuple[int, int],
            num_groups: int,
            input_channels: int,
            output_channels: int,
            bias: bool,
            stride: int,
            dilation: int,
    ):
        self.weight, self.bias = get_conv_weight_and_bias(kernel_size, num_groups, input_channels, output_channels, bias)
        self.groups = num_groups
        self.stride = stride
        self.dilation = dilation

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, in_height, in_width = x.size()
        kernel_height, kernel_width = self.kernel_size

        dilated_kernel_height = (kernel_height - 1) * self.dilation + 1
        dilated_kernel_width = (kernel_width - 1) * self.dilation + 1

        out_height = (in_height - dilated_kernel_height) // self.stride + 1
        out_width = (in_width - dilated_kernel_width) // self.stride + 1

        output = torch.zeros((batch_size, self.output_channels, out_height, out_width), device=x.device)

        in_channels_per_group = in_channels // self.groups
        out_channels_per_group = self.output_channels // self.groups

        for batch in range(batch_size):
            for g in range(self.groups):
                for out_c in range(out_channels_per_group):
                    output_channel_idx = g * out_channels_per_group + out_c
                    for i in range(out_height):
                        for j in range(out_width):
                            # Compute the starting indices for the sliding window
                            h_start = i * self.stride
                            w_start = j * self.stride
                            
                            # Apply dilation in filter placement
                            h_end = h_start + dilated_kernel_height
                            w_end = w_start + dilated_kernel_width

                            # Select the portion of the input tensor
                            input_slice = x[batch, g * in_channels_per_group:(g + 1) * in_channels_per_group, h_start:h_end:self.dilation, w_start:w_end:self.dilation]
                            
                            # Apply the convolution filter
                            output[batch, output_channel_idx, i, j] = torch.sum(input_slice * self.weight[output_channel_idx]) 
                            
                            # Add bias if it's enabled
                            if self.bias is not None:
                                output[batch, output_channel_idx, i, j] += self.bias[output_channel_idx]

        
        return output


class MyFilterStub:
    def __init__(
            self,
            filter: torch.Tensor,
            input_channels: int,
    ):
        self.weight = filter
        self.input_channels = input_channels
        self.kernel_size = filter.shape


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, in_height, in_width = x.shape
        out_height = in_height - self.kernel_size[0] + 1
        out_width = in_width - self.kernel_size[1] + 1
        filter_height, filter_width = self.weight.shape
        output = torch.zeros((batch_size, in_channels, out_height, out_width))

        for b in range(batch_size):
            for c in range(in_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        region = x[b, c, i:i + filter_height, j:j + filter_width]
                        output[b, c, i, j] = torch.sum(region * self.weight)
    
        return output
