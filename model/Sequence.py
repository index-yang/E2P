import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torch.optim as optim
import torchvision.models as models


class SequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_len):
        super(SequenceModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout = 0.3)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout = 0.3)

        self.fc = nn.Linear(hidden_size, 5 * output_size)  # 每个时间步输出一个类别（如 0, 1, 2）
        
        self.seq_len = seq_len  # 序列长度

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 初始化隐藏状态
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 初始化单元状态
        # LSTM 前向传播
        embedding_out, _ = self.lstm(x, (h_0, c_0))  
        
        # out, _ = self.gru(x)
        out = self.fc(embedding_out)  # 线性层将隐藏状态转换为输出类别
        # out = torch.clamp(out, min=0,max=4)
        return embedding_out,out  # 输出为 (batch_size, seq_len, output_size)，即每个时间步的输出类别

