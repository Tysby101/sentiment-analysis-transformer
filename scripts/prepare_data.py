"""
Prepare IMDB dataset for training.
"""
from datasets import load_from_disk
from pathlib import Path
from src.utils.vocab import Vocabulary
import pickle

def prepare_data():
    # Load dataset
    print("📂 Loading IMDB dataset...")
    dataset = load_from_disk("data/imdb")
    
    # Build vocabulary from training data
    vocab = Vocabulary(max_vocab_size=10000)
    train_texts = dataset['train']['text']
    vocab.build_from_texts(train_texts)
    
    # Save vocabulary
    vocab_path = Path("data/vocab.pkl")
    vocab.save(str(vocab_path))
    
    # Show statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    print(f"Vocabulary size: {len(vocab)}")
    
    # Sample encoding
    sample_text = dataset['train'][0]['text']
    encoded = vocab.encode(sample_text)
    print(f"\n📝 Sample encoding:")
    print(f"Original: {sample_text[:100]}...")
    print(f"Encoded: {encoded[:20]}...")
    print(f"Decoded: {vocab.decode(encoded[:20])}...")

if __name__ == "__main__":
    prepare_data()