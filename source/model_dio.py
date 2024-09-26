import math

import numpy as np
import torch
import torch.nn as nn

from source.model_resnet1d import Bottleneck1D, BasicBlock1D


class DIO(nn.Module):

    def __init__(self):
        super(DIO, self).__init__()

        group_sizes = [2, 2, 2, 2]
        base_plane = 64
        kernel_size = 3

        self.inplanes = base_plane

        # Input module
        self.input_block = nn.Sequential(
            nn.Conv1d(6, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Residual groups
        planes = [base_plane * (2 ** i) for i in range(len(group_sizes))]
        strides = [1] + [2] * (len(group_sizes) - 1)
        dilations = [1] * len(group_sizes)
        groups = [self._make_residual_group1d(BasicBlock1D, planes[i], kernel_size, group_sizes[i], strides[i], dilations[i]) for i in range(len(group_sizes))]
        self.residual_groups = nn.Sequential(*groups)

        # Output module
        self.output_block = nn.Conv1d(512, 128, kernel_size=1, bias=False)

        self.out_v = nn.Sequential(
            nn.Linear(896, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.ReLU(True),
        )

        self.out_t = nn.Sequential(
            nn.Linear(896, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
        )

        self._initialize(False)

    def _make_residual_group1d(self, block_type, planes, kernel_size, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_type.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block_type.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block_type.expansion))
        layers = []
        layers.append(block_type(self.inplanes, planes, kernel_size=kernel_size,
                                 stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block_type.expansion
        for _ in range(1, blocks):
            layers.append(block_type(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _initialize(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck1D):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock1D):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x_input_block = self.input_block(x)

        x_residual_block = self.residual_groups(x_input_block)

        # return x_residual_block

        x_output_block = self.output_block(x_residual_block)

        x_flatten = x_output_block.view(x_output_block.size(0), -1)

        out_v = self.out_v(x_flatten)
        out_t = self.out_t(x_flatten)

        return out_v, out_t

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
