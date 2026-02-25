"""
Training script for sentiment analysis.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from pathlib import Path
from tqdm import tqdm
import math

from src.models.classifier import SentimentClassifier
from data.dataset import SentimentDataset
from src.utils.vocab import Vocabulary
from config import Config

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Learning rate scheduler with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (inputs, masks, labels) in enumerate(pbar):
        inputs = inputs.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        logits = model(inputs, masks)
        loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    return total_loss / len(dataloader), 100. * correct / total

def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation/test set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, masks, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            logits = model(inputs, masks)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total

def main():
    # Config
    config = Config()
    
    # Device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Load vocabulary
    vocab = Vocabulary.load(config.vocab_path)
    config.vocab_size = len(vocab)
    print(f"📚 Vocabulary size: {config.vocab_size}")
    
    # Load data
    print("📂 Loading dataset...")
    dataset = load_from_disk(config.data_dir)
    
    # Create datasets
    train_dataset = SentimentDataset(
        texts=dataset['train']['text'],
        labels=dataset['train']['label'],
        vocab=vocab,
        max_len=config.max_len
    )
    
    test_dataset = SentimentDataset(
        texts=dataset['test']['text'],
        labels=dataset['test']['label'],
        vocab=vocab,
        max_len=config.max_len
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Test batches: {len(test_loader)}")
    
    # Create model
    model = SentimentClassifier(
        vocab_size=config.vocab_size,
        num_classes=config.num_classes,
        d_model=config.d_model,
        num_layers=config.num_layers,
        heads=config.heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        max_len=config.max_len
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🔢 Total parameters: {total_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    num_training_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Create checkpoint directory
    Path(config.checkpoint_dir).mkdir(exist_ok=True)
    
    # Training loop
    best_acc = 0
    patience = 3
    patience_counter = 0
    print(f"\n🚀 Starting training for {config.num_epochs} epochs...\n")
    
    for epoch in range(config.num_epochs):
        print(f"📍 Epoch {epoch + 1}/{config.num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, config
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%\n")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'config': config
            }, f"{config.checkpoint_dir}/best_model.pt")
            print(f"💾 Saved best model (acc: {best_acc:.2f}%)\n")
        else:
            patience_counter += 1
            print(f"⚠️  No improvement for {patience_counter} epoch(s)")
            
            if patience_counter >= patience:
                print("🛑 Early stopping triggered!")
                break
    
    print(f"✅ Training complete! Best test accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()