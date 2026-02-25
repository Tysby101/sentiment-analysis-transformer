from src.components.encoder_layer import EncoderLayer
from src.components.positional import PositionalEncoding
import torch.nn as nn
import math

class TransformerEncoder(nn.Module):
    """Implementation of TransformerEncoder"""
    def __init__(self, vocab_size=8000, num_layers=6, d_model=256, heads=8, d_ff=512, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.heads = heads
        self.d_ff = d_ff
        self.dropout = dropout

        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional Encoder
        self.positional_encoder = PositionalEncoding(d_model=d_model)

        # N encoder layers
        self.layers = nn.ModuleList([ EncoderLayer(d_model=d_model, heads=heads, d_ff=d_ff, dropout=dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len]
        """

        # Embedding projection
        emb = self.embedding(x)*math.sqrt(self.d_model) # scale it to avoid Positional encoding from over domination

        # Positional enconding
        x = self.positional_encoder(emb)

        # Go through layers
        for layer in self.layers:
            x, _ = layer(x, mask)

        return x