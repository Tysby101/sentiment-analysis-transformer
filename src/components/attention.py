import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """This is the ScaledDotProductAttention"""
    def __init__(self, temperature=None):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: [batch_size, heads, seq_len, d_k]
            K: [batch_size, heads, seq_len, d_k]
            V: [batch_size, heads, seq_len, d_k]
        """

        # Step 1: compute the dot product
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) # [batch_size, heads, seq_len, seq_len]

        # Step 2: apply temperature is set
        if self.temperature is not None:
            attention_scores = attention_scores/self.temperature

        # Step 3: apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e-9)

        # Apply softmax to get weights
        attention_weights = F.softmax(attention_scores, dim=-1) # [batch_size, heads, seq_len, seq_len]

        # Compute output
        output = torch.matmul(attention_weights, V) # [batch_size, heads, seq_len, d_k]

        return output, attention_weights