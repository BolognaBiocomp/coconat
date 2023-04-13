import torch
import torch.nn as nn

class MeanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
                        nn.Linear(2312, 64),
                        nn.Dropout(p=0.1),
                        nn.ReLU(),
                        nn.Linear(64, 4)
                        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        return x
