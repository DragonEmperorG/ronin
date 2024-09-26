import math

import numpy as np
import torch
import torch.nn as nn


class TAHNN(nn.Module):

    def __init__(self):
        super(TAHNN, self).__init__()

        self.cnn_conv1d = nn.Conv1d(6, 64, 7, stride=2, padding=3, bias=False)
        self.cnn_batchnorm1d = nn.BatchNorm1d(64, affine=False)
        self.cnn_relu = nn.ReLU()
        self.cnn_maxpool1d = nn.MaxPool1d(3, stride=2, padding=1)

        self.lstm = nn.LSTM(64, 120, num_layers=2, batch_first=True, bidirectional=True)

        self.tam_conv1d = nn.Conv1d(49, 64, 49)
        self.tam_final_hidden_linear = nn.Linear(240, 64)
        self.tam_sigmoid = nn.Sigmoid()
        self.tam_final_c_linear = nn.Linear(64 + 240, 240, bias=False)

        self.out_v_linear1 = nn.Linear(240, 256)
        self.out_v_linear2 = nn.Linear(256, 1)
        self.out_v_relu = nn.ReLU()
        self.out_t_linear1 = nn.Linear(240, 256)
        self.out_t_linear2 = nn.Linear(256, 2)

    def forward(self, x):
        x_cnn_permute = x.permute(0, 2, 1)

        x_cnn_conv1d = self.cnn_conv1d(x_cnn_permute)
        x_cnn_batchnorm1d = self.cnn_batchnorm1d(x_cnn_conv1d)
        x_cnn_relu = self.cnn_relu(x_cnn_batchnorm1d)
        x_cnn_maxpool1d = self.cnn_maxpool1d(x_cnn_relu)

        x_lstm_permute = x_cnn_maxpool1d.permute(0, 2, 1)
        x_lstm = self.lstm(x_lstm_permute)
        x_lstm_output = x_lstm[0]

        x_tam_prev = x_lstm_output[:, 0:-1, :]
        x_tam_prev_conv1d = self.tam_conv1d(x_tam_prev)

        x_tam_final = x_lstm_output[:, -1, :]
        x_tam_final_linear = self.tam_final_hidden_linear(x_tam_final)

        x_tam_K = x_tam_prev_conv1d.permute(0, 2, 1)
        x_tam_h = x_tam_final_linear.permute(1, 0)
        x_tam_score_func = torch.matmul(x_tam_K, x_tam_h)
        x_tam_alpha = self.tam_sigmoid(x_tam_score_func)
        x_tam_dot = x_tam_K * x_tam_alpha
        x_tam_sum = torch.sum(x_tam_dot, dim=1)

        x_hnnta_feature = torch.cat((x_tam_sum, x_tam_final), 1)
        x_out_c = self.tam_final_c_linear(x_hnnta_feature)

        x_out_v_linear1 = self.out_v_linear1(x_out_c)
        x_out_v_linear2 = self.out_v_linear2(x_out_v_linear1)
        x_out_v = self.out_v_relu(x_out_v_linear2)

        x_out_t_linear1 = self.out_t_linear1(x_out_c)
        x_out_t = self.out_t_linear2(x_out_t_linear1)

        return x_out_v, x_out_t
