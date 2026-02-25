"""
PyTorch Dataset for sentiment analysis.
"""
import torch
from torch.utils.data import Dataset
from typing import List, Tuple

class SentimentDataset(Dataset):
    """Dataset for sentiment analysis"""
    
    def __init__(self, texts: List[str], labels: List[int], vocab, max_len: int = 256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get text and label
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Encode text
        indices = self.vocab.encode(text)
        
        # Truncate or pad
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        
        # Create mask (1 for real tokens, 0 for padding)
        mask_len = len(indices)
        
        # Pad to max_len
        padding_len = self.max_len - len(indices)
        indices = indices + [self.vocab.word2idx[self.vocab.PAD_TOKEN]] * padding_len
        
        # Create attention mask
        mask = [1] * mask_len + [0] * padding_len
        
        return (
            torch.tensor(indices, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )