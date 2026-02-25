"""
Download and prepare IMDB dataset for sentiment analysis.
"""
from datasets import load_dataset
import os
from pathlib import Path

def download_imdb():
    """Download IMDB dataset from HuggingFace"""
    print("📥 Downloading IMDB dataset...")
    
    # Load dataset
    dataset = load_dataset("imdb")
    
    # Create data directory
    data_dir = Path("data/imdb")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to disk
    dataset.save_to_disk(str(data_dir))
    
    print(f"✅ Dataset saved to {data_dir}")
    print(f"Train examples: {len(dataset['train'])}")
    print(f"Test examples: {len(dataset['test'])}")
    
    # Show sample
    print("\n📝 Sample review:")
    sample = dataset['train'][0]
    print(f"Text: {sample['text'][:200]}...")
    print(f"Label: {sample['label']} (0=negative, 1=positive)")
    
    return dataset

if __name__ == "__main__":
    download_imdb()