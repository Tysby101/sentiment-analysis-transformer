"""
Vocabulary builder for text tokenization.
"""
from collections import Counter
from typing import List
import pickle


class Vocabulary:
    """Simple word-level vocabulary"""
    
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    
    def __init__(self, max_vocab_size: int = 10000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
        }
        self.idx2word = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}

    def build_from_texts(self, texts: List[str]):
        """Build vocabulary from list of texts"""
        print("🔨 Building vocabulary...")
        
        # Count words
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)

        # Take most common words
        most_common = word_counts.most_common(self.max_vocab_size - 2)

        # Add to vocab
        for idx, (word, count) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"✅ Vocabulary built: {len(self.word2idx)} words")
        print(f"Most common: {most_common[:10]}")

    def encode(self, text: str) -> List[int]:
        """Convert text to list of indices"""
        words = text.lower().split()
        return [self.word2idx.get(word, self.word2idx[self.UNK_TOKEN]) for word in words]
    
    def decode(self, indices: List[int]) -> str:
        """Convert indices back to text"""
        words = [self.idx2word.get(idx, self.UNK_TOKEN) for idx in indices]
        return " ".join(words)
    
    def __len__(self):
        return len(self.word2idx)
    
    def save(self, path: str):
        """Save vocabulary to disk"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'max_vocab_size': self.max_vocab_size
            }, f)
        print(f"💾 Vocabulary saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Load vocabulary from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        vocab = cls(max_vocab_size=data['max_vocab_size'])
        vocab.word2idx = data['word2idx']
        vocab.idx2word = data['idx2word']
        print(f"📂 Vocabulary loaded from {path}")
        return vocab
    