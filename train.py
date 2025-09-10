"""
Training script for Transformer model on Finnish-English translation
Advanced NLP Assignment 1
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import GradScaler, autocast

import numpy as np
import argparse
import os
import time
import json
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import copy
import math

from decoder import EncoderDecoderTransformer
from utils import NoamScheduler, LabelSmoothingLoss, ModelConfig


class Vocabulary:
    """Vocabulary for tokenization and encoding."""
    
    def __init__(self, freq_threshold: int = 2):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
        
    def build_vocabulary(self, sentences: List[str]):
        """Build vocabulary from sentences."""
        frequencies = Counter()
        idx = 4  # Start after special tokens
        
        for sentence in sentences:
            for word in sentence.split():
                frequencies[word] += 1
        
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
    
    def numericalize(self, text: str) -> List[int]:
        """Convert text to numerical representation."""
        tokenized = text.split()
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized]
    
    def __len__(self):
        return len(self.itos)


class TranslationDataset(Dataset):
    """Dataset for Finnish-English translation."""
    
    def __init__(self, src_file: str, tgt_file: str, 
                 src_vocab: Optional[Vocabulary] = None,
                 tgt_vocab: Optional[Vocabulary] = None,
                 max_length: int = 100):
        self.max_length = max_length
        
        # Read files
        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_sentences = [line.strip().lower() for line in f]
        
        with open(tgt_file, 'r', encoding='utf-8') as f:
            self.tgt_sentences = [line.strip().lower() for line in f]
        
        assert len(self.src_sentences) == len(self.tgt_sentences), \
            "Source and target files must have the same number of lines"
        
        # Build or use provided vocabularies
        if src_vocab is None:
            self.src_vocab = Vocabulary()
            self.src_vocab.build_vocabulary(self.src_sentences)
        else:
            self.src_vocab = src_vocab
            
        if tgt_vocab is None:
            self.tgt_vocab = Vocabulary()
            self.tgt_vocab.build_vocabulary(self.tgt_sentences)
        else:
            self.tgt_vocab = tgt_vocab
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_text = self.src_sentences[idx]
        tgt_text = self.tgt_sentences[idx]
        
        # Numericalize
        src_indices = self.src_vocab.numericalize(src_text)
        tgt_indices = self.tgt_vocab.numericalize(tgt_text)
        
        # Truncate if necessary
        src_indices = src_indices[:self.max_length]
        tgt_indices = tgt_indices[:self.max_length-2]  # Leave room for SOS/EOS
        
        # Add SOS and EOS tokens to target (teacher forcing setup)
        tgt_indices = [self.tgt_vocab.stoi["<SOS>"]] + tgt_indices + [self.tgt_vocab.stoi["<EOS>"]]
        
        return torch.tensor(src_indices), torch.tensor(tgt_indices)


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    src_batch, tgt_batch = [], []
    
    for src, tgt in batch:
        src_batch.append(src)
        tgt_batch.append(tgt)
    
    # Pad sequences
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    
    return src_batch, tgt_batch


class TransformerTrainer:
    """Trainer class for Transformer model."""
    
    def __init__(self, model: EncoderDecoderTransformer, 
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=1.0,  # NoamScheduler will scale this
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=1e-4  # Add L2 regularization
        )
        
        # Learning rate schedulers
        self.scheduler = NoamScheduler(
            self.optimizer, 
            d_model=model.encoder.d_model,
            warmup_steps=8000
        )
        
        # Add ReduceLROnPlateau as backup when Noam schedule isn't enough
        self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True,
            min_lr=1e-6
        )
        
        # Loss function - Using Label Smoothing from utils
        self.criterion = LabelSmoothingLoss(
            vocab_size=model.decoder.vocab_size,
            padding_idx=0,
            smoothing=0.1
        )
        
        # Mixed precision training
        self.scaler = GradScaler() if args.use_amp else None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_perplexities = []
        self.learning_rates = []
        self.gradient_norms = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_perplexity = float('inf')
        self.best_epoch = 1
        self.best_model_state = None
        
        # Early stopping parameters
        self.patience = args.early_stopping_patience
        self.patience_counter = 0
        self.early_stop = False
        self.min_delta = args.min_delta
        
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch. Returns (avg_loss, avg_grad_norm)"""
        self.model.train()
        total_loss = 0
        grad_norms = []  # Store all finite gradient norms
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (src, tgt) in enumerate(progress_bar):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            # Teacher forcing: use ground truth as input
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.args.use_amp and self.scaler is not None:
                with autocast():
                    output = self.model(src, tgt_input)
                    loss = self.criterion(output, tgt_output)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"NaN loss detected at batch {batch_idx}, skipping batch")
                    continue
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping before unscaling
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.clip_grad
                )
                
                # Only store finite gradient norms
                if torch.isfinite(grad_norm):
                    grad_norms.append(grad_norm.item())
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                output = self.model(src, tgt_input)
                loss = self.criterion(output, tgt_output)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"NaN loss detected at batch {batch_idx}, skipping batch")
                    continue
                
                loss.backward()
                
                # Gradient clipping and get norm
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.clip_grad
                )
                
                # Only store finite gradient norms
                if torch.isfinite(grad_norm):
                    grad_norms.append(grad_norm.item())
                
                self.optimizer.step()
            
            # Step the Noam scheduler after each batch
            self.scheduler.step()
            current_lr = self.scheduler.get_lr()
            
            total_loss += loss.item()
            
            # Update progress bar with safe gradient norm display
            display_grad_norm = grad_norm.item() if torch.isfinite(grad_norm) else -1
            progress_bar.set_postfix({
                'loss': loss.item(), 
                'lr': f'{current_lr:.6f}',
                'grad_norm': f'{display_grad_norm:.2f}' if display_grad_norm >= 0 else 'inf'
            })
            
            # Log detailed metrics every N batches
            if batch_idx % self.args.log_interval == 0 and batch_idx > 0:
                avg_loss = total_loss / (batch_idx + 1)
                # Calculate average of finite gradient norms only
                if grad_norms:
                    avg_grad_norm = np.mean(grad_norms)
                    print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                          f'Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, '
                          f'Avg Grad Norm: {avg_grad_norm:.2f} '
                          f'({len(grad_norms)} finite gradients)')
                else:
                    print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                          f'Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, '
                          f'Avg Grad Norm: N/A (no finite gradients)')
        
        # Calculate final averages
        avg_loss = total_loss / len(self.train_loader)
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
        
        return avg_loss, avg_grad_norm
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model. Returns (avg_loss, perplexity)"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for src, tgt in tqdm(self.val_loader, desc='Validating'):
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                output = self.model(src, tgt_input)
                loss = self.criterion(output, tgt_output)
                
                if not torch.isnan(loss):
                    total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow
        
        return avg_loss, perplexity
    
    def train(self):
        """Main training loop with early stopping."""
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Using Noam scheduler with warmup_steps=8000")
        print(f"Using Label Smoothing Loss with smoothing=0.1")
        print(f"Weight decay: {self.optimizer.param_groups[0]['weight_decay']}")
        print(f"Dropout rate: {self.args.dropout}")
        print(f"Gradient clipping: {self.args.clip_grad}")
        print(f"Early stopping patience: {self.patience} epochs")
        print(f"Minimum delta for improvement: {self.min_delta}")
        
        for epoch in range(1, self.args.epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{self.args.epochs}")
            print(f"{'='*50}")
            
            # Training
            train_loss, avg_grad_norm = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.gradient_norms.append(avg_grad_norm)
            
            # Validation
            val_loss, val_perplexity = self.validate()
            self.val_losses.append(val_loss)
            self.val_perplexities.append(val_perplexity)
            
            # Step the plateau scheduler with validation loss
            self.plateau_scheduler.step(val_loss)
            
            # Store current learning rate
            current_lr = self.scheduler.get_lr()
            self.learning_rates.append(current_lr)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Val Perplexity: {val_perplexity:.2f}")
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"Average Gradient Norm: {avg_grad_norm:.2f}")
            
            # Check if validation loss improved
            if val_loss < self.best_val_loss - self.min_delta:
                print(f"Validation loss improved from {self.best_val_loss:.4f} to {val_loss:.4f}")
                
                # Always save the best validation loss model
                self.best_val_loss = val_loss
                self.best_val_perplexity = val_perplexity
                self.patience_counter = 0
                self.best_epoch = epoch
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                print(f"New best model saved! Val loss: {val_loss:.4f}, Perplexity: {val_perplexity:.2f}")
                
                # Warn if perplexity seems too low
                if val_perplexity < 2.0:
                    print(f"WARNING: Very low perplexity ({val_perplexity:.2f}) may indicate overfitting")
            else:
                self.patience_counter += 1
                print(f"No improvement. Patience: {self.patience_counter}/{self.patience}")
            
            # Early stopping check
            if self.patience_counter >= self.patience:
                print(f"\n{'='*50}")
                print(f"EARLY STOPPING TRIGGERED!")
                print(f"No improvement for {self.patience} consecutive epochs")
                print(f"Best validation loss: {self.best_val_loss:.4f}")
                print(f"Best validation perplexity: {self.best_val_perplexity:.2f}")
                print(f"Best epoch was: {self.best_epoch}")
                print(f"Stopping at epoch {epoch}")
                print(f"{'='*50}")
                
                self.early_stop = True
                break
            
            # Additional stopping criterion: if train loss is too low compared to val loss
            if train_loss < 0.5 and val_loss > train_loss * 10:
                print(f"\n{'='*50}")
                print(f"STOPPING DUE TO OVERFITTING!")
                print(f"Train loss ({train_loss:.4f}) << Val loss ({val_loss:.4f})")
                print(f"{'='*50}")
                self.early_stop = True
                break
            
            # Plot training curves
            if epoch % self.args.plot_interval == 0:
                self.plot_training_curves()
        
        # Save the best model
        self.save_best_model()
        
        # Plot final training curves
        self.plot_training_curves(final=True)
    
    def save_best_model(self):
        """Save the best model state."""
        print(f"\n{'='*50}")
        if self.early_stop:
            print("Training stopped early")
        else:
            print("Training completed all epochs")
        
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation perplexity: {self.best_val_perplexity:.2f}")
        print(f"Best epoch: {self.best_epoch}")
        print("Saving best model...")
        
        if self.best_model_state is not None:
            # Create model config
            model_config = ModelConfig(
                src_vocab_size=len(self.train_loader.dataset.src_vocab),
                tgt_vocab_size=len(self.train_loader.dataset.tgt_vocab),
                d_model=self.args.d_model,
                n_heads=self.args.n_heads,
                n_encoder_layers=self.args.n_layers,
                n_decoder_layers=self.args.n_layers,
                d_ff=self.args.d_ff,
                max_seq_len=self.args.max_seq_len,
                dropout=self.args.dropout,
                pos_encoding_type=self.args.pos_encoding,
                pad_idx=0
            )
            
            checkpoint = {
                'epoch': self.best_epoch,
                'model_state_dict': self.best_model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_perplexities': self.val_perplexities,
                'learning_rates': self.learning_rates,
                'gradient_norms': self.gradient_norms,
                'best_val_loss': self.best_val_loss,
                'best_val_perplexity': self.best_val_perplexity,
                'args': self.args,
                'model_config': model_config.to_dict(),
                'vocab_path': os.path.join(self.args.save_dir, 'vocabs.pkl'),
                'early_stopped': self.early_stop,
                'patience_counter': self.patience_counter
            }
            
            path = os.path.join(self.args.save_dir, f'best_model_{self.args.pos_encoding}.pt')
            torch.save(checkpoint, path)
            print(f"Best model saved to {path}")
        print(f"{'='*50}")
    
    def plot_training_curves(self, final: bool = False):
        """Plot and save training curves."""
        plt.figure(figsize=(20, 5))
        
        # Loss plot
        plt.subplot(1, 4, 1)
        plt.plot(self.train_losses, label='Train Loss', marker='o', markersize=3)
        plt.plot(self.val_losses, label='Validation Loss', marker='s', markersize=3)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Curves - {self.args.pos_encoding.upper()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add best point marker
        if self.val_losses:
            best_epoch_idx = np.argmin(self.val_losses)
            plt.plot(best_epoch_idx, self.val_losses[best_epoch_idx], 'r*', markersize=15, 
                    label=f'Best: {self.val_losses[best_epoch_idx]:.4f}')
        
        # Perplexity plot
        plt.subplot(1, 4, 2)
        if self.val_perplexities:
            plt.plot(self.val_perplexities, marker='o', markersize=3, color='green')
            plt.axhline(y=self.best_val_perplexity, color='r', linestyle='--', 
                       label=f'Best: {self.best_val_perplexity:.2f}')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Perplexity')
        plt.title('Validation Perplexity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate plot
        plt.subplot(1, 4, 3)
        if self.learning_rates:
            plt.plot(self.learning_rates, marker='o', markersize=3)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        
        # Gradient norm plot
        plt.subplot(1, 4, 4)
        if self.gradient_norms:
            # Filter out any remaining non-finite values for plotting
            finite_grad_norms = [g for g in self.gradient_norms if np.isfinite(g)]
            if finite_grad_norms:
                plt.plot(finite_grad_norms, marker='o', markersize=3, color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Average Gradient Norm')
        plt.title('Gradient Norms')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        suffix = '_final' if final else ''
        plt.savefig(os.path.join(self.args.save_dir, 
                                 f'training_curves_{self.args.pos_encoding}{suffix}.png'))
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Transformer for Finnish-English Translation')
    
    # Data arguments
    parser.add_argument('--src_file', type=str, default='EUbookshop.fi',
                       help='Source (Finnish) file')
    parser.add_argument('--tgt_file', type=str, default='EUbookshop.en',
                       help='Target (English) file')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1,
                       help='Test split ratio')
    parser.add_argument('--data_fraction', type=float, default=1.0,
                       help='Fraction of data to use (for quick testing)')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=512,
                       help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6,
                       help='Number of encoder/decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048,
                       help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.3,  # Increased from 0.1
                       help='Dropout rate')
    parser.add_argument('--max_seq_len', type=int, default=100,
                       help='Maximum sequence length')
    parser.add_argument('--pos_encoding', type=str, default='rope',
                       choices=['rope', 'relative'],
                       help='Positional encoding type')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,  # Reduced from 50
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate (note: Noam scheduler will adjust this)')
    parser.add_argument('--clip_grad', type=float, default=0.5,
                       help='Gradient clipping')
    parser.add_argument('--use_amp', action='store_true',
                       help='Use mixed precision training')
    
    # Early stopping arguments
    parser.add_argument('--early_stopping_patience', type=int, default=5,  # Reduced from 10
                       help='Early stopping patience (epochs)')
    parser.add_argument('--min_delta', type=float, default=0.01,  # Increased from 0.001
                       help='Minimum change in validation loss to qualify as improvement')
    
    # Logging arguments
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Log every N batches')
    parser.add_argument('--plot_interval', type=int, default=1,
                       help='Plot training curves every N epochs')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.save_dir, f'args_{args.pos_encoding}.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("Loading and preparing data...")
    
    # Load all data
    with open(args.src_file, 'r', encoding='utf-8') as f:
        all_src = f.readlines()
    with open(args.tgt_file, 'r', encoding='utf-8') as f:
        all_tgt = f.readlines()
    
    # Use only a fraction of the data if specified
    if args.data_fraction < 1.0:
        n_samples = int(len(all_src) * args.data_fraction)
        all_src = all_src[:n_samples]
        all_tgt = all_tgt[:n_samples]
        print(f"Using {args.data_fraction*100:.1f}% of data: {n_samples} samples")
    
    # Calculate split indices
    n_total = len(all_src)
    n_test = int(n_total * args.test_split)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_test - n_val
    
    # Split data
    train_src = all_src[:n_train]
    train_tgt = all_tgt[:n_train]
    val_src = all_src[n_train:n_train+n_val]
    val_tgt = all_tgt[n_train:n_train+n_val]
    test_src = all_src[n_train+n_val:]
    test_tgt = all_tgt[n_train+n_val:]
    
    # Save splits to files
    for split_name, src_data, tgt_data in [
        ('train', train_src, train_tgt),
        ('val', val_src, val_tgt),
        ('test', test_src, test_tgt)
    ]:
        with open(f'{split_name}.fi', 'w', encoding='utf-8') as f:
            f.writelines(src_data)
        with open(f'{split_name}.en', 'w', encoding='utf-8') as f:
            f.writelines(tgt_data)
    
    print(f"Data split: Train={n_train}, Val={n_val}, Test={n_test}")
    
    # Create datasets
    train_dataset = TranslationDataset('train.fi', 'train.en', max_length=args.max_seq_len)
    val_dataset = TranslationDataset('val.fi', 'val.en',
                                    src_vocab=train_dataset.src_vocab,
                                    tgt_vocab=train_dataset.tgt_vocab,
                                    max_length=args.max_seq_len)
    
    # Save vocabularies
    with open(os.path.join(args.save_dir, 'vocabs.pkl'), 'wb') as f:
        pickle.dump({
            'src_vocab': train_dataset.src_vocab,
            'tgt_vocab': train_dataset.tgt_vocab
        }, f)
    
    print(f"Source vocabulary size: {len(train_dataset.src_vocab)}")
    print(f"Target vocabulary size: {len(train_dataset.tgt_vocab)}")
    print(f"Special tokens: PAD=0, SOS=1, EOS=2, UNK=3")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    model = EncoderDecoderTransformer(
        src_vocab_size=len(train_dataset.src_vocab),
        tgt_vocab_size=len(train_dataset.tgt_vocab),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_layers,
        n_decoder_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        pos_encoding_type=args.pos_encoding,
        pad_idx=0
    )
    
    # Create trainer and train
    trainer = TransformerTrainer(model, train_loader, val_loader, args)
    trainer.train()
    
    print("\nTraining completed!")
    if trainer.early_stop:
        print(f"Model stopped early at epoch {len(trainer.train_losses)}")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Best validation perplexity: {trainer.best_val_perplexity:.2f}")
    
    # Evaluate on test set after training
    if os.path.exists('test.fi') and os.path.exists('test.en'):
        print("\n" + "="*50)
        print("Evaluating on test set...")
        print("="*50)
        
        from test import TranslationEvaluator, DecodingConfig
        
        # Load best model
        best_checkpoint_path = os.path.join(args.save_dir, f'best_model_{args.pos_encoding}.pt')
        if os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(best_checkpoint_path, map_location=trainer.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Create test dataset
            test_dataset = TranslationDataset('test.fi', 'test.en',
                                            src_vocab=train_dataset.src_vocab,
                                            tgt_vocab=train_dataset.tgt_vocab,
                                            max_length=args.max_seq_len)
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=2
            )
            
            # Evaluate with greedy decoding
            evaluator = TranslationEvaluator(model, test_loader, 
                                           train_dataset.src_vocab, 
                                           train_dataset.tgt_vocab, 
                                           trainer.device)
            
            config = DecodingConfig(max_length=args.max_seq_len)
            result = evaluator.evaluate('greedy', config)
            
            print(f"\nFINAL TEST SET BLEU SCORE: {result['bleu']:.2f}")
            print("="*50)


if __name__ == '__main__':
    main()