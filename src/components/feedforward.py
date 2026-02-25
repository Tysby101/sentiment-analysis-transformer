
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """Implementation of FeedForward layer"""
    def __init__(self, d_model = 256, d_ff = 512, dropout=0.1):
        super(FeedForward, self).__init__()

        # Two linear and relu
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.activation(self.linear1(x))
        x = self.dropout(x)

        return self.linear2(x)