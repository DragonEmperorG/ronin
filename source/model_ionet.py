import torch
import torch.nn as nn


class IONet(nn.Module):

    def __init__(self):
        super(IONet, self).__init__()
        self.fc_lstm_input = nn.Linear(in_features=6, out_features=128, bias=False)
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=128, bias=False, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=256, bias=False, bidirectional=True)
        self.fc_lstm_output = nn.Linear(in_features=512, out_features=256, bias=False)
        self.fc_polar_vector = nn.Linear(in_features=256, out_features=2, bias=False)
        self.fc_uncertainty = nn.Linear(in_features=256, out_features=2, bias=False)

    def forward(self, x):
        lstm1_input = self.fc_lstm_input(x)
        lstm1_output = self.lstm1(lstm1_input)
        lstm2_input = lstm1_output[0]
        lstm2_output = self.lstm2(lstm2_input)
        lstm_output = lstm2_output[0][-1, :]
        fc_input = self.fc_lstm_output(lstm_output)
        polar_vector = self.fc_polar_vector(fc_input)
        uncertainty = self.fc_uncertainty(fc_input)
        return polar_vector, uncertainty


class IONetLSTM(nn.Module):

    def __init__(self, hidden_size, num_layers):
        super(IONetLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=hidden_size, num_layers=num_layers, bias=False, batch_first=True)
        self.fc_polar_vector = nn.Linear(in_features=hidden_size, out_features=2, bias=False)
        self.fc_uncertainty = nn.Linear(in_features=hidden_size, out_features=2, bias=False)

    def forward(self, x):
        lstm_output = self.lstm(x)
        fc_input = lstm_output[0][:, -1, :]
        polar_vector = self.fc_polar_vector(fc_input)
        uncertainty = self.fc_uncertainty(fc_input)
        return polar_vector, uncertainty


class IONetGRU(nn.Module):

    def __init__(self, hidden_size, num_layers):
        super(IONetGRU, self).__init__()
        self.gru = nn.GRU(input_size=6, hidden_size=hidden_size, num_layers=num_layers, bias=False, batch_first=True)
        self.fc_polar_vector = nn.Linear(in_features=hidden_size, out_features=2, bias=False)
        self.fc_uncertainty = nn.Linear(in_features=hidden_size, out_features=2, bias=False)

    def forward(self, x):
        gru_output = self.gru(x)
        fc_input = gru_output[0][:, -1, :]
        polar_vector = self.fc_polar_vector(fc_input)
        uncertainty = self.fc_uncertainty(fc_input)
        return polar_vector, uncertainty
