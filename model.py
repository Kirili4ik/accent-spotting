import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchaudio
import os
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import numpy as np

class TSC(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, n_groups=1, 
                 dilation=1):
        super(TSC, self).__init__()
        self.tsc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size, 
                      dilation=dilation, groups=in_channels,
                      padding=dilation * (kernel_size  - 1) // 2 ),
            nn.Conv1d(in_channels, out_channels, 1, groups=n_groups),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        x = self.tsc(x)
        return x  


class TSCActivated(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, n_groups=1, 
                 dilation=1):
        super(TSCActivated, self).__init__()
        self.tsc = TSC(kernel_size, in_channels, out_channels, n_groups, 
                       dilation)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.tsc(x)
        x = self.activation(x)
        return x  


class TSCBlock(nn.Module):
    def __init__(self, n_blocks, kernel_size, in_channels, out_channels,
                 n_groups=1, is_intermediate=False):
        super(TSCBlock, self).__init__()
        if is_intermediate:
            in_channels = out_channels
        self.n_blocks = n_blocks
        self.tsc_list = nn.ModuleList([TSCActivated(kernel_size, in_channels, out_channels, n_groups)])
        self.tsc_list.extend([TSCActivated(kernel_size, out_channels, out_channels, n_groups) 
                                  for i in range(1, self.n_blocks-1)])
        self.tsc_list.append(TSC(kernel_size, out_channels, out_channels, n_groups))
        self.pnt_wise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, groups=n_groups)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x_res = self.bn(self.pnt_wise_conv(x))
        for layer in self.tsc_list:
            x = layer(x)
        return self.relu(x + x_res)


class ConvBlock(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, dilation=1, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, 
                      padding=dilation * (kernel_size - 1) // 2, dilation=dilation, 
                      stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Debug(nn.Module):
    def __init__(self, msg=''):
        super().__init__()
        self.msg = msg
    
    def forward(self, x):
        print(f'{x.shape}\n{self.msg}')
        return x


class QuarzNet(nn.Module):
    def __init__(self, config):
        super().__init__() 
        self.config = config
        self.net = nn.Sequential(
            TSCActivated(**config['c1']),
            TSCBlock(**config['b1']),
            TSCBlock(**config['b2']),
            TSCBlock(**config['b3']),
            TSCBlock(**config['b4']),
            TSCBlock(**config['b5']),
            TSCActivated(**config['c2']),
            Debug(),
            TSCActivated(**config['c3']),
            Debug()
        )

    def forward(self, x):
        x = self.net(x)
        return x


class ClassificationNet(nn.Module):
    
    def __init__(self, config):
        super().__init__() 
        self.encoder = QuarzNet(config)
        hidden_size, attn_size = config['hidden_size'], config['attn_size']
        n_classes = config['n_classes']
        self.attention_scores = nn.Sequential(
            nn.Linear(hidden_size, attn_size),
            nn.Tanh(),
            nn.Linear(attn_size, 1, bias=False),
            nn.Softmax(dim=1)
        )
        self.out = nn.Linear(hidden_size, n_classes)
        
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = x.transpose(1, 2)
        attn_weights = self.attn(x)
        x = torch.bmm(attn_weights.transpose(1, 2), x)
        out = self.out(x)
        return out.squeeze(1), attention_weights


def make_param_dict(names, params):
    param_dict = {n : p for n, p in zip(names, params)}
    return param_dict


def make_config():
    n_mels = 80
    n_classes = 34
    config = {
        #  k, in, out, dilation
        'c1': [33, n_mels, 128, 1],
        'c2': [87, 512, 512, 2],
        'c3': [1, 512, 512, 1],
        # n_blocks, k, in, out
        'b1': [5, 33, 128, 128],
        'b2': [5, 39, 128, 256],
        'b3': [5, 51, 256, 256],
        'b4': [5, 63, 256, 512],
        'b5': [5, 75, 512, 512],
        'hidden_size': 512,
        'attn_size': 256,
        'n_classes' : n_classes,
        'n_epochs': 25,
        'n_mels': n_mels,
        'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    }

    return config

