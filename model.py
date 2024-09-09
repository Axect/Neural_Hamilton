import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, hparams, device='cpu'):
        super(MLP, self).__init__()
        self.hparams = hparams
        self.device  = device
        
        nodes   = hparams['nodes']
        layers  = hparams['layers']
        input_size  = 1
        output_size = 1

        net = [nn.Linear(input_size, nodes), nn.GELU()]
        for _ in range(layers-1):
            net.append(nn.Linear(nodes, nodes))
            net.append(nn.GELU())
        net.append(nn.Linear(nodes, output_size))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
