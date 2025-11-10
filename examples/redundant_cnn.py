import torch
import torch.nn as nn
from examples import consts

class RedundantCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(consts.IN_CHANNELS, consts.HIDDEN_CHANNELS, consts.KERNEL_SIZE)
        self.conv2 = nn.Conv2d(consts.HIDDEN_CHANNELS, consts.OUT_CHANNELS, consts.KERNEL_SIZE)
        self.bn = nn.BatchNorm2d(consts.BN_CHANNELS)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(consts.FLATTENED_SIZE, consts.LINEAR_HIDDEN)
        self.linear2 = nn.Linear(consts.LINEAR_HIDDEN, consts.LINEAR_OUTPUT)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = x + 0.0000000001 #Redundant Code
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        temp = torch.tensor(3.1415) + torch.tensor(2.718)
        _ = temp * 1
        _ = temp + 0
        _ = x * 0
        x = self.relu(x)
        return x

def get_model():
    return RedundantCNN().eval(), torch.randn(
        consts.BATCH_SIZE, consts.IN_CHANNELS, consts.INPUT_HEIGHT, consts.INPUT_WIDTH
    )
