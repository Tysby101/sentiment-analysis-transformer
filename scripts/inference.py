"""
Make predictions on new text.
"""
import torch
from src.models.classifier import SentimentClassifier
from src.utils.vocab import Vocabulary
from config import Config

def predict(text: str, model, vocab, device, max_len=256):
    """Predict sentiment for a single text"""
    model.eval()
    
    # Encode text
    indices = vocab.encode(text)
    
    # Truncate or pad
    if len(indices) > max_len:
        indices = indices[:max_len]
    
    mask_len = len(indices)
    padding_len = max_len - len(indices)
    indices = indices + [vocab.word2idx[vocab.PAD_TOKEN]] * padding_len
    mask = [1] * mask_len + [0] * padding_len
    
    # Convert to tensors
    inputs = torch.tensor([indices], dtype=torch.long).to(device)
    masks = torch.tensor([mask], dtype=torch.long).to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(inputs, masks)
        probs = torch.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1).item()
    
    sentiment = "Positive" if pred == 1 else "Negative"
    confidence = probs[0][pred].item()
    
    return sentiment, confidence

def main():
    # Load config and vocab
    config = Config()
    vocab = Vocabulary.load(config.vocab_path)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = SentimentClassifier(
        vocab_size=len(vocab),
        num_classes=config.num_classes,
        d_model=config.d_model,
        num_layers=config.num_layers,
        heads=config.heads,
        d_ff=config.d_ff,
        dropout=0.0,  # No dropout during inference
        max_len=config.max_len
    ).to(device)
    
    checkpoint = torch.load(f"{config.checkpoint_dir}/best_model.pt", map_location=device, weights_only=False )
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval() # eval mode
    
    print(f"✅ Model loaded (Test Acc: {checkpoint['test_acc']:.2f}%)\n")
    
    # Example predictions
    test_texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible film, waste of time. Would not recommend.",
        "It was okay, nothing special but not bad either.",
        "One of the best films I've ever seen! Masterpiece!",
        "Boring and predictable. Very disappointed."
    ]
    
    print("🎬 Predictions:\n")
    for text in test_texts:
        sentiment, confidence = predict(text, model, vocab, device, config.max_len)
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment} (confidence: {confidence:.2%})\n")

if __name__ == "__main__":
    main()