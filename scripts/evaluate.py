"""
Detailed evaluation with metrics and confusion matrix.
"""
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
from torch.utils.data import DataLoader

from models.classifier import SentimentClassifier
from data.dataset import SentimentDataset
from utils.vocab import Vocabulary
from config import Config

def plot_confusion_matrix(cm, save_path="confusion_matrix.png"):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive']
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Confusion matrix saved to {save_path}")

def evaluate_model():
    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    # Load vocabulary and data
    vocab = Vocabulary.load(config.vocab_path)
    dataset = load_from_disk(config.data_dir)
    
    test_dataset = SentimentDataset(
        texts=dataset['test']['text'],
        labels=dataset['test']['label'],
        vocab=vocab,
        max_len=config.max_len
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Load model
    model = SentimentClassifier(
        vocab_size=len(vocab),
        num_classes=config.num_classes,
        d_model=config.d_model,
        num_layers=config.num_layers,
        heads=config.heads,
        d_ff=config.d_ff,
        dropout=0.0,
        max_len=config.max_len
    ).to(device)
    

    checkpoint = torch.load(f"./{config.checkpoint_dir}/best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded from epoch {checkpoint['epoch']} (Test Acc: {checkpoint['test_acc']:.2f}%)\n")
    
    # Collect predictions
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("🔍 Evaluating on test set...")
    with torch.no_grad():
        for inputs, masks, labels in test_loader:
            inputs = inputs.to(device)
            masks = masks.to(device)
            
            logits = model(inputs, masks)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Classification report
    print("\n📈 Classification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=['Negative', 'Positive'],
        digits=4
    ))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\n📊 Confusion Matrix:")
    print(cm)
    plot_confusion_matrix(cm)
    
    # Confidence analysis
    correct_mask = all_preds == all_labels
    correct_probs = all_probs[correct_mask].max(axis=1)
    incorrect_probs = all_probs[~correct_mask].max(axis=1)
    
    print(f"\n🎯 Confidence Analysis:")
    print(f"Avg confidence (correct): {correct_probs.mean():.4f}")
    print(f"Avg confidence (incorrect): {incorrect_probs.mean():.4f}")
    
    # Show some errors
    print("\n❌ Sample Misclassifications:")
    test_texts = dataset['test']['text']
    errors = np.where(~correct_mask)[0][:5]
    
    for idx in errors:
        text = test_texts[idx]
        true_label = "Positive" if all_labels[idx] == 1 else "Negative"
        pred_label = "Positive" if all_preds[idx] == 1 else "Negative"
        confidence = all_probs[idx].max()
        
        print(f"\nText: {text[:150]}...")
        print(f"True: {true_label} | Predicted: {pred_label} (conf: {confidence:.2%})")

if __name__ == "__main__":
    evaluate_model()