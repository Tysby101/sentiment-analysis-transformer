"""
Training configuration.
"""
from dataclasses import dataclass

@dataclass
class Config:
    # Model
    d_model: int = 256
    num_layers: int = 2
    heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_len: int = 256
    
    # Data
    vocab_size: int = 10000
    num_classes: int = 2
    
    # Training
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 5e-4
    weight_decay: float = 0.05
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Paths
    data_dir: str = "data/imdb"
    vocab_path: str = "data/vocab.pkl"
    checkpoint_dir: str = "checkpoints"
    
    # Device
    device: str = "cuda"  # or "cpu" or "mps"
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 1000