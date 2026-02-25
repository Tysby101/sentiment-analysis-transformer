import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Implementation of Sinusoidal Positional encoding"""
    def __init__(self, d_model = 256, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.max_len = max_len

        # Create an empty pe
        pe = torch.zeros((max_len, d_model))

        # Create position
        positions = torch.arange(0, max_len).unsqueeze(-1) # unsqueeze to allow broadcasting

        # Compute div term
        div_term = torch.exp(-torch.arange(0, d_model, 2)*(math.log(10000.0)/d_model) ) # 1/(10000**(2i/d_model))

        # Fill the pe
        pe[:, 0::2] = torch.sin(positions*div_term) # even indices 2i
        pe[:, 1::2] = torch.cos(positions*div_term) # even indices 2i+1

        # register
        self.register_buffer("pe", pe)


    def forward(self, x):
        """
        Args:
            [batch_size, seq_len, d_model]
        """

        return x + self.pe[:x.size(1),:]