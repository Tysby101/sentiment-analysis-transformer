from src.components.feedforward import FeedForward
from src.components.multihead import MultiHeadAttention
import torch.nn as nn

class EncoderLayer(nn.Module):
    """Implementation of EncoderLayer"""
    def __init__(self, d_model=256, heads=8, d_ff=512, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.feedforward = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.attention = MultiHeadAttention(heads=heads, d_model=d_model)

    def forward(self, x, mask=None):
        """
        doubt about two or one dropout
        Args:
            x: [batch_size, seq_len, d_model]
        """
        # Attention: here
        attn_out, attn_weights = self.attention(x, x, x, mask)

        # Add & norm
        x = self.norm1(x + self.dropout1(attn_out))

        residual = x

        # Feed Forward
        x = self.feedforward(x)

        # Add & norm
        x = self.norm2(residual + self.dropout2(x))
        
        return x, attn_weights