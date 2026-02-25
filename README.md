# sentiment-analysis-transformer
A complete Transformer encoder implementation for sentiment analysis on IMDB movie reviews, built entirely from scratch using PyTorch.
# Sentiment Analysis with Transformer (From Scratch)

A complete Transformer encoder implementation for sentiment analysis on IMDB movie reviews, built entirely from scratch using PyTorch. No pretrained models or transformer libraries—every component hand-coded.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **83.37%** |
| **Precision** | 83.41% |
| **F1-Score** | 83.37% |
| **Training Time** | ~18s/epoch (GPU) |
| **Model Parameters** | ~8.5M |
| **Dataset** | IMDB (50k reviews) |

## 🏗️ Architecture

All components implemented from scratch:

1. **Scaled Dot-Product Attention**
   - `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
   - Temperature scaling, masking support

2. **Multi-Head Attention** (8 heads)
   - Parallel attention, separate Q/K/V projections
   - Concatenation + final projection

3. **Positional Encoding**
   - Sinusoidal embeddings: `sin/cos(pos/10000^(2i/d_model))`
   - No learnable parameters

4. **Feed-Forward Network**
   - Two linear layers + ReLU + dropout
   - Dimension: 256 → 1024 → 256

5. **Encoder Layer**
   - Multi-head self-attention + residual + LayerNorm
   - Feed-forward + residual + LayerNorm

6. **Full Stack**
   - 4 encoder layers, mean pooling, classification head

## 📊 Configuration

```python
d_model = 256       # Hidden dimension
num_layers = 4      # Encoder layers  
heads = 4           # Attention heads
d_ff = 512         # FFN dimension
dropout = 0.1       # Dropout rate
vocab_size = 10000  # Vocabulary size
max_len = 256       # Max sequence length
```

## 🚀 Quick Start

```bash
# Install dependencies
uv pip install -e .

# Prepare data
python data/download_imdb.py
python scripts/prepare_data.py

# Train
python scripts/train.py

# Inference
python scripts/inference.py

# Evaluation
python scripts/evaluate.py
```

## 📈 Results

### Training Progress
| Epoch | Train Acc | Test Acc | Status |
|-------|-----------|----------|--------|
| 5 | 85.63% | 82.97% | |
| 6 | 87.34% | 82.69% | |
| **7** | **89.21%** | **83.37%** | ✓ Best |
| 8 | 91.17% | 82.30% | Overfitting |
| 10 | 96.36% | 82.54% | Early stop |

### Classification Report
```
              precision    recall  f1-score   support
    Negative     0.8447    0.8178    0.8310     12500
    Positive     0.8234    0.8497    0.8363     12500
    accuracy                         0.8337     25000
```

### Sample Misclassifications

**False Positive (90.52% confidence):**
```
Text: "First off let me say, If you haven't enjoyed a Van Damme 
       movie since bloodsport, you probably will not like this movie..."
True: Negative | Predicted: Positive
```

**False Positive (78.09% confidence):**
```
Text: "Ben is a deeply unhappy adolescent, the son of his 
       unhappily married parents..."
True: Negative | Predicted: Positive
```

## 🎓 Key Learnings

1. **Attention**: Q/K/V mechanism with learned projections (even in self-attention, Q≠K≠V due to different weight matrices)
2. **Positional Encoding**: Critical for sequence order; embeddings scaled by √d_model to prevent PE dominance
3. **Residual Connections**: Enable gradient flow in deep networks
4. **Overfitting**: Model memorizes after epoch 7 (train 89%→96%, test 83%→82%); early stopping essential

## 🔬 Future Improvements

- [ ] Subword tokenization (BPE/WordPiece) for better vocab coverage
- [ ] Data augmentation (word drop, synonym replacement)
- [ ] CLS token pooling instead of mean pooling
- [ ] Implement Decoder for seq2seq tasks

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper

## 🤝 Author

**Cheick Tidiani Cissé**
- GitHub: [@tysby101](https://github.com/Tysby101)
- LinkedIn: [Cheick Tidiani Cissé, Ph.D](https://www.linkedin.com/in/cheick-tidiani-ciss%C3%A9-ph-d-83364613a/)

---

⭐ **Star this repo** if you found it helpful!