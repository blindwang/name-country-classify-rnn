import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        batch_size, seq_len, input_size = inputs.size()  # [batch_size, n_seq, n_letters]
        inputs = inputs.permute(1, 0, 2)  # 调整输入张量的维度为[seq_len, batch_size, input_size]
        hidden = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        outputs = []
        for i in range(seq_len):
            input = inputs[i]
            combined = torch.cat((input, hidden), -1)  # [B, input+hidden]
            hidden = self.i2h(combined)
            output = self.i2o(combined)
            output = torch.log_softmax(output, dim=-1)
            # output = self.softmax(output)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)  # 将输出张量的维度调整为[batch_size, n_seq, output_size]
        return outputs[:,-1,:]


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(BiRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden_size, output_size)

    def forward(self, inputs):
        outputs, h_n = self.rnn(inputs)
        outputs = self.fc(outputs[:, -1, :])
        outputs = torch.log_softmax(outputs, dim=-1)
        return outputs


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()

        self.hidden_size = hidden_size

        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.new_memory = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        batch_size, seq_len, input_size = inputs.size()  # [batch_size, n_seq, n_letters]
        inputs = inputs.permute(1, 0, 2)  # 调整输入张量的维度为[seq_len, batch_size, input_size]
        hidden = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        outputs = []
        for i in range(seq_len):
            input = inputs[i]
            combined = torch.cat((input, hidden), -1)
            update_gate = torch.sigmoid(self.update_gate(combined))
            reset_gate = torch.sigmoid(self.reset_gate(combined))
            new_memory = torch.tanh(self.new_memory(torch.cat((input, reset_gate * hidden), -1)))
            hidden = (1-update_gate) * hidden + update_gate * new_memory
            output = self.output_layer(hidden)
            output = self.softmax(output)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)  # 将输出张量的维度调整为[batch_size, n_seq, output_size]
        return outputs[:, -1, :]


class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(BiGRU, self).__init__()

        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden_size, output_size)

    def forward(self, inputs):
        outputs, h_n = self.gru(inputs)
        outputs = self.fc(outputs[:, -1, :])
        outputs = torch.log_softmax(outputs, dim=-1)
        return outputs


class LSTM(nn.Module): 
    def __init__(self, input_size, hidden_size, output_size): 
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size

        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        batch_size, seq_len, input_size = inputs.size()  # [batch_size, n_seq, n_letters]
        inputs = inputs.permute(1, 0, 2)  # 调整输入张量的维度为[seq_len, batch_size, input_size]
        hidden = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        cell = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        outputs = []
        for i in range(seq_len):
            input = inputs[i]
            combined = torch.cat((input, hidden), -1)
            forget_gate = torch.sigmoid(self.forget_gate(combined))
            input_gate = torch.sigmoid(self.input_gate(combined))
            cell_gate = torch.tanh(self.cell_gate(combined))
            cell = forget_gate * cell + input_gate * cell_gate
            output_gate = torch.sigmoid(self.output_gate(combined))
            hidden = output_gate * torch.tanh(cell)
            output = self.output_layer(hidden)
            output = self.softmax(output)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)  # 将输出张量的维度调整为[batch_size, n_seq, output_size]
        return outputs[:, -1, :]

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(BiLSTM, self).__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_size, output_size)

    def forward(self, inputs):
        outputs, h_n = self.lstm(inputs)
        outputs = self.fc(outputs[:, -1, :])
        outputs = torch.log_softmax(outputs, dim=-1)
        return outputs
