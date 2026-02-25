from src.components.attention import ScaledDotProductAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Implementation of MultiHeadAttention"""
    def __init__(self, heads=8, d_model=256):
        super(MultiHeadAttention, self).__init__()

        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model//heads

        # Projection layer
        self.Q_W = nn.Linear(d_model, d_model)
        self.K_W = nn.Linear(d_model, d_model)
        self.V_W = nn.Linear(d_model, d_model)

        # Attention layer
        self.attention = ScaledDotProductAttention(temperature=math.sqrt(self.d_k))

        # Final projection layer
        self.FP_W = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: [batch_size, seq_len, d_model]
            K: [batch_size, seq_len, d_model]
            V: [batch_size, seq_len, d_model]
        """
        batch_size = Q.size(0)
        seq_len = Q.size(1)

        # Project first
        Q_proj = self.Q_W(Q)
        K_proj = self.K_W(K)
        V_proj = self.V_W(V)

        # Split heads and Transpose
        Q_proj = Q_proj.view(batch_size, seq_len, self.heads, self.d_k).transpose(2, 1) # [batch_size, heads, seq_len, d_k]
        K_proj = K_proj.view(batch_size, seq_len, self.heads, self.d_k).transpose(2, 1) # [batch_size, heads, seq_len, d_k]
        V_proj = V_proj.view(batch_size, seq_len, self.heads, self.d_k).transpose(2, 1) # [batch_size, heads, seq_len, d_k]

        # Apply attention
        attention_output, attention_weights = self.attention(Q_proj, K_proj, V_proj, mask) # [batch_size, heads, seq_len, d_k]

        # Transpose and Reshape back
        attention_output = attention_output.transpose(2, 1).contiguous().view(batch_size, seq_len, self.d_model) # [batch_size, seq_len, d_model]

        # Last projection layer
        out = self.FP_W(attention_output) # [batch_size, seq_len, d_model]

        return out, attention_weights