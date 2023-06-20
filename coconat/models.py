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

class TransposeX(nn.Module):
    def __init__(self):
        super(TransposeX, self).__init__()
    def forward(self, x):
        return torch.transpose(x, -2, -1)

class MMModelLSTM(nn.Module):
    def __init__(self,
                IN_SIZE: int = 2304,
                OUT_CHANNELS: int = 40,
                KERNEL_SIZE: int = 15,
                LSTM_HIDDEN: int = 128,
                HIDDEN_DIM: int = 64,
                NUM_LSTM: int = 1,
                DROPOUT: float = 0.25,
                OUT_SIZE: int = 8):
        super().__init__()
        self.cnn = nn.Sequential(
        TransposeX(),
        nn.Conv1d(
            IN_SIZE,
            OUT_CHANNELS,
            KERNEL_SIZE,
            padding='same'
            ),
            TransposeX(),
            nn.Dropout(p=DROPOUT)
        )

        self.lstm = nn.LSTM(
            OUT_CHANNELS,
            LSTM_HIDDEN,
            num_layers=NUM_LSTM,
            batch_first=True)

        self.linear = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, HIDDEN_DIM),
            nn.Dropout(p=DROPOUT),
            nn.ReLU(),
            TransposeX(),
            nn.BatchNorm1d(HIDDEN_DIM),
            TransposeX(),
            nn.Linear(HIDDEN_DIM, OUT_SIZE)
        )
        self.final = nn.Softmax(dim=-1)

    def forward(self, x, l):
        x = self.cnn(x)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, l, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = self.final(self.linear(x))
        return x
