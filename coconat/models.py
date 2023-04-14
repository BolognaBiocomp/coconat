import torch
import torch.nn as nn

class MeanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
                        nn.Linear(4608, 128),
                        nn.Dropout(p=0.1),
                        nn.ReLU(),
                        nn.Linear(128, 4)
                        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.softmax(self.linear(x))
        return x
