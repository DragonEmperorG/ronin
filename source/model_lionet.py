import math

import numpy as np
import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """
    Input and output sizes will be the same.
    #[in_size+2*pad-dilation*(kernel_size-1)-1]/stride+1
    if stride ==1:
    then
        out_size = in_size+2*pad-dilation*(kernel_size-1)
    finially:
        out_size = in_size
    recommand:
        stride = 1
        kernel_size % 2 =1
    """
    def __init__(self, in_size, out_size, kernel_size, stride=1, dilation=1, bias=True):
        super(CausalConv1d, self).__init__()
        self.pad = dilation * (kernel_size - 1)
        self.conv1 = nn.Conv1d(in_size, out_size, kernel_size, padding=self.pad, stride=stride, dilation=dilation, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        if self.pad > 0:
            x = x[..., :-self.pad]
        return x


class ResidualLayer(nn.Module):
    def __init__(self, residual_size, skip_size, dilation):
        super(ResidualLayer, self).__init__()
        self.conv_filter = CausalConv1d(residual_size, residual_size, kernel_size=2, dilation=dilation)
        self.conv_gate = CausalConv1d(residual_size, residual_size, kernel_size=2, dilation=dilation)
        self.resconv1_1 = nn.Conv1d(residual_size, residual_size, kernel_size=1)
        self.skipconv1_1 = nn.Conv1d(residual_size, skip_size, kernel_size=1)

    def forward(self, x):
        conv_filter = self.conv_filter(x)
        conv_gate = self.conv_gate(x)
        fx = torch.tanh(conv_filter) * torch.sigmoid(conv_gate)
        fx = self.resconv1_1(fx)
        skip = self.skipconv1_1(fx)
        residual = fx + x
        # residual=[batch,residual_size,seq_len]  skip=[batch,skip_size,seq_len]
        return skip, residual


class DilatedStack(nn.Module):
    def __init__(self, residual_size, skip_size, dilation_depth):
        super(DilatedStack, self).__init__()
        residual_stack = [ResidualLayer(residual_size, skip_size, 2 ** layer) for layer in range(dilation_depth)]
        self.residual_stack = nn.ModuleList(residual_stack)

    def forward(self, x):
        skips = []
        for layer in self.residual_stack:
            skip, x = layer(x)
            skips.append(skip.unsqueeze(0))
            # skip =[1,batch,skip_size,seq_len]
        return torch.cat(skips, dim=0), x  # [layers,batch,skip_size,seq_len]


class LIONet(nn.Module):

    def __init__(self):
        super(LIONet, self).__init__()
        self.input_size = 6
        self.residual_size = 32
        self.skip_size = 32
        self.dilation_cycles = 1
        self.dilation_depth = 8

        self.input_conv = CausalConv1d(self.input_size, self.residual_size, kernel_size=1)

        self.dilated_stacks = nn.ModuleList(
            [DilatedStack(self.residual_size, self.skip_size, self.dilation_depth) for cycle in range(self.dilation_cycles)]
        )

        self.convout_1 = nn.Conv1d(self.skip_size, self.skip_size, kernel_size=1)
        self.convout_2 = nn.Conv1d(self.skip_size, self.skip_size, kernel_size=1)

        self.output_pool = nn.AvgPool1d(200)
        self.output_fc = nn.Linear(self.skip_size, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, input_feature_dim, seq_len] = [N, C_in, Lin]

        x = self.input_conv(x)  # [batch, residual_size, seq_len]

        skip_connections = []

        for cycle in self.dilated_stacks:
            skips, x = cycle(x)
            skip_connections.append(skips)

        ## skip_connection=[total_layers,batch,skip_size,seq_len]
        skip_connections = torch.cat(skip_connections, dim=0)

        # gather all output skip connections to generate output, discard last residual output

        out = skip_connections.sum(dim=0)  # [batch,skip_size,seq_len]

        out = nn.functional.relu(out)
        out = self.convout_1(out)  # [batch,out_size,seq_len]

        out = nn.functional.relu(out)
        out = self.convout_2(out)

        out = self.output_pool(out)
        out = torch.squeeze(out, 2)

        out = self.output_fc(out)

        return out
