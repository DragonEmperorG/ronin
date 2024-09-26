import os
import time
from os import path as osp

import numpy as np
import torch
import json

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tensorboardX import SummaryWriter
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import DataLoader

from data_glob_speed import *
from source.model_dio import DIO
from source.model_hnnta import TAHNN
from source.model_ionet import IONetLSTM, IONetGRU, IONet
from source.model_lionet import LIONet
from source.model_temporal import LSTMSeqNetwork, TCNSeqNetwork
from transformations import *
from metric import compute_ate_rte
from model_resnet1d import *

from ptflops import get_model_complexity_info

from thop import profile
from thop import clever_format
from torchinfo import summary

_input_channel, _output_channel = 6, 2
_fc_config = {'fc_dim': 512, 'in_dim': 7, 'dropout': 0.5, 'trans_planes': 128}


def get_model(arch):
    if arch == 'resnet18':
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [2, 2, 2, 2],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
    elif arch == 'resnet50':
        # For 1D network, the Bottleneck structure results in 2x more parameters, therefore we stick to BasicBlock.
        _fc_config['fc_dim'] = 1024
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [3, 4, 6, 3],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
    elif arch == 'resnet101':
        _fc_config['fc_dim'] = 1024
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [3, 4, 23, 3],
                           base_plane=64, output_block=FCOutputModule, **_fc_config)
    elif arch == 'roninlstm':
        network = LSTMSeqNetwork(_input_channel, _output_channel, 1, 'cuda:0', lstm_layers=3, lstm_size=100)
    elif arch == 'ronintcn':
        network = TCNSeqNetwork(_input_channel, _output_channel, 3, layer_channels=[16, 32, 64, 128, 72, 36])
    elif arch == 'ionet':
        network = IONet()
    elif arch == 'ionetlstm1l64h':
        network = IONetLSTM(64, 1)
    elif arch == 'ionetlstm1l128h':
        network = IONetLSTM(128, 1)
    elif arch == 'ionetlstm2l128h':
        network = IONetLSTM(128, 2)
    elif arch == 'ionetgru1l128h':
        network = IONetGRU(128, 1)
    elif arch == 'lionet':
        network = LIONet()
    elif arch == 'dio':
        network = DIO()
    elif arch == 'hnnta':
        network = TAHNN()
    else:
        raise ValueError('Invalid architecture: ', args.arch)
    return network


def model_info(args, **kwargs):
    # Loading model
    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')
    network = get_model(args.arch)
    # macs, params = get_model_complexity_info(network, (200, 6), as_strings=False, backend='pytorch', print_per_layer_stat=True, verbose=True, output_precision=9)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # thop_input = torch.randn(1, 200, 6)
    # macs, params, ret_dict = profile(network, inputs=(thop_input,), ret_layer_info=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    summary(network, input_size=(1, 6, 200), depth=10)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()

    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})

    model_info(args)
