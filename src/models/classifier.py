import torch
import torch.nn as nn
from src.components.transformer_encoder import TransformerEncoder


class SentimentClassifier(nn.Module):
    """Transformer-based sentiment classifier"""
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 2,
        d_model: int = 256,
        num_layers: int = 4,
        heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 256
    ):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            num_layers=num_layers,
            d_model=d_model,
            heads=heads,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len] - token indices
            mask: [batch_size, seq_len] - attention mask (1=real, 0=padding)
        Returns:
            logits: [batch_size, num_classes]
        """
        # Reshape mask for multi-head attention: [batch, 1, 1, seq_len]
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
        
        # Encode: [batch_size, seq_len, d_model]
        encoded = self.encoder(x, mask)
        
        # Mean pooling over sequence (ignoring padding)
        if mask is not None:
            mask_expanded = mask.squeeze(1).squeeze(1).unsqueeze(-1)  # [batch, seq_len, 1]
            sum_encoded = (encoded * mask_expanded).sum(dim=1)  # [batch, d_model]
            mask_sum = mask_expanded.sum(dim=1)  # [batch, 1]
            pooled = sum_encoded / mask_sum.clamp(min=1)  # Avoid division by zero
        else:
            pooled = encoded.mean(dim=1)  # [batch, d_model]
        
        # Classify
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # [batch, num_classes]
        
        return logits