"""
Utility functions for Transformer implementation
Advanced NLP Assignment 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Any
import json
import os
from dataclasses import dataclass, asdict
import sacrebleu
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from collections import defaultdict
import pandas as pd
import pickle
import time


class NoamScheduler:
    """
    Noam learning rate scheduler as used in "Attention is All You Need".
    Implements the formula from the paper:
    lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
    """
    
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
        # Set initial learning rate
        self._update_learning_rate()
        
    def step(self):
        """Update learning rate based on current step."""
        self.current_step += 1
        self._update_learning_rate()
    
    def _update_learning_rate(self):
        """Update the learning rate in the optimizer."""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self):
        """Calculate learning rate for current step using the paper's formula."""
        step = max(1, self.current_step)
        # From the paper: d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
        scale = self.d_model ** (-0.5)
        
        if step < self.warmup_steps:
            # Linear warmup
            return scale * step * (self.warmup_steps ** (-1.5))
        else:
            # Inverse square root decay
            return scale * (step ** (-0.5))
    
    def state_dict(self):
        """Save scheduler state."""
        return {'current_step': self.current_step}
    
    def load_state_dict(self, state_dict):
        """Load scheduler state."""
        self.current_step = state_dict['current_step']
        self._update_learning_rate()


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for better generalization.
    Implements the technique from the original Transformer paper with epsilon=0.1
    """
    
    def __init__(self, vocab_size: int, padding_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate label smoothing loss.
        
        Args:
            logits: Model predictions (batch_size * seq_len, vocab_size) or (batch_size, seq_len, vocab_size)
            targets: Target indices (batch_size * seq_len) or (batch_size, seq_len)
        """
        # Reshape to (batch_size * seq_len, vocab_size)
        if logits.dim() == 3:
            batch_size, seq_len, vocab_size = logits.size()
            logits = logits.reshape(-1, vocab_size)
        else:
            vocab_size = logits.size(-1)
            
        if targets.dim() == 2:
            targets = targets.reshape(-1)
        
        # Create smoothed target distribution
        with torch.no_grad():
            # Create uniform distribution over vocabulary (excluding padding)
            true_dist = torch.zeros_like(logits)
            # Distribute smoothing mass over all tokens except padding
            # Subtract 1 to exclude padding token from denominator
            true_dist.fill_(self.smoothing / (vocab_size - 1))
            
            # Put confidence mass on the true labels
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
            
            # Zero out padding positions completely
            padding_mask = targets == self.padding_idx
            true_dist[padding_mask] = 0
        
        # Calculate log probabilities with numerical stability
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Calculate KL divergence loss
        loss = -torch.sum(true_dist * log_probs, dim=-1)
        
        # Mask out padding tokens from loss
        non_pad_mask = ~padding_mask
        loss = loss * non_pad_mask.float()
        
        # Return average loss over non-padding tokens
        num_non_pad = non_pad_mask.sum().float()
        # Prevent division by zero
        num_non_pad = torch.clamp(num_non_pad, min=1.0)
        
        return loss.sum() / num_non_pad


def initialize_weights(model: nn.Module):
    """
    Initialize model weights using Xavier uniform initialization.
    Based on the initialization scheme from the original Transformer paper.
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_attention_weights(attention_weights: torch.Tensor, 
                          save_path: str,
                          src_tokens: Optional[List[str]] = None,
                          tgt_tokens: Optional[List[str]] = None):
    """
    Visualize and save attention weights heatmap.
    
    Args:
        attention_weights: Attention weights tensor (heads, tgt_len, src_len)
        save_path: Path to save the figure
        src_tokens: Source sequence tokens for labels
        tgt_tokens: Target sequence tokens for labels
    """
    # Average over heads
    if len(attention_weights.shape) == 3:
        attention_weights = attention_weights.mean(dim=0)
    
    attention_weights = attention_weights.cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, cmap='Blues', cbar=True, 
                xticklabels=src_tokens if src_tokens else False,
                yticklabels=tgt_tokens if tgt_tokens else False)
    plt.xlabel('Source Position')
    plt.ylabel('Target Position')
    plt.title('Attention Weights Visualization')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def sacre_bleu_from_pairs(predictions: List[str], references: List[List[str]]) -> float:
    """
    Calculate BLEU score using sacrebleu with proper reference transposition.
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations (each item is a list of possible references)
    
    Returns:
        BLEU score as float
    """
    # Transpose references for sacrebleu format
    # From: [[ref1_for_pred1], [ref1_for_pred2], ...]
    # To: [[ref1_for_pred1, ref1_for_pred2, ...]]
    refs_transposed = list(map(list, zip(*references)))
    bleu = sacrebleu.corpus_bleu(predictions, refs_transposed)
    return bleu.score


class TranslationMetrics:
    """Calculate various translation metrics."""
    
    @staticmethod
    def calculate_bleu(predictions: List[str], 
                       references: List[List[str]],
                       use_sacrebleu: bool = True) -> float:
        """
        Calculate BLEU score.
        
        Args:
            predictions: List of predicted translations
            references: List of reference translations (can be multiple per prediction)
            use_sacrebleu: Whether to use sacrebleu (more standard) or NLTK
        """
        if use_sacrebleu:
            return sacre_bleu_from_pairs(predictions, references)
        else:
            # NLTK BLEU
            pred_tokens = [pred.split() for pred in predictions]
            ref_tokens = [[ref.split() for ref in ref_list] for ref_list in references]
            return corpus_bleu(ref_tokens, pred_tokens) * 100
    
    @staticmethod
    def calculate_sentence_bleu(prediction: str, 
                               references: List[str]) -> float:
        """Calculate BLEU score for a single sentence."""
        pred_tokens = prediction.split()
        ref_tokens = [ref.split() for ref in references]
        return sentence_bleu(ref_tokens, pred_tokens) * 100


class DataStatistics:
    """Calculate and display dataset statistics."""
    
    @staticmethod
    def calculate_stats(file_path: str, vocab: Any = None) -> Dict:
        """
        Calculate statistics for a data file.
        
        Returns:
            Dictionary with statistics
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        lengths = [len(line.split()) for line in lines]
        
        stats = {
            'num_sentences': len(lines),
            'avg_length': np.mean(lengths),
            'max_length': max(lengths),
            'min_length': min(lengths),
            'std_length': np.std(lengths),
            'percentile_95': np.percentile(lengths, 95),
            'percentile_99': np.percentile(lengths, 99)
        }
        
        if vocab:
            stats['vocab_size'] = len(vocab)
            stats['unk_tokens'] = sum(1 for line in lines 
                                     for token in line.split() 
                                     if token not in vocab.stoi)
        
        return stats
    
    @staticmethod
    def plot_length_distribution(src_file: str, tgt_file: str, save_path: str):
        """Plot length distribution of source and target sequences."""
        with open(src_file, 'r', encoding='utf-8') as f:
            src_lengths = [len(line.split()) for line in f]
        
        with open(tgt_file, 'r', encoding='utf-8') as f:
            tgt_lengths = [len(line.split()) for line in f]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Source distribution
        axes[0].hist(src_lengths, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Sequence Length')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Source (Finnish) Length Distribution')
        axes[0].axvline(np.mean(src_lengths), color='red', 
                       linestyle='--', label=f'Mean: {np.mean(src_lengths):.1f}')
        axes[0].legend()
        
        # Target distribution
        axes[1].hist(tgt_lengths, bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Sequence Length')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Target (English) Length Distribution')
        axes[1].axvline(np.mean(tgt_lengths), color='red', 
                       linestyle='--', label=f'Mean: {np.mean(tgt_lengths):.1f}')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


class ExperimentTracker:
    """Track and log experiment results."""
    
    def __init__(self, experiment_name: str, save_dir: str = 'experiments'):
        self.experiment_name = experiment_name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.metrics = defaultdict(list)
        self.config = {}
        
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a single metric value."""
        if step is not None:
            self.metrics[name].append((step, value))
        else:
            self.metrics[name].append(value)
    
    def log_config(self, config: Dict):
        """Log experiment configuration."""
        self.config = config
    
    def save(self):
        """Save experiment data to file."""
        data = {
            'experiment_name': self.experiment_name,
            'config': self.config,
            'metrics': dict(self.metrics)
        }
        
        # Save as JSON
        json_path = os.path.join(self.save_dir, f'{self.experiment_name}.json')
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save as pickle for complex objects
        pkl_path = os.path.join(self.save_dir, f'{self.experiment_name}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
    
    def plot_metrics(self):
        """Plot all logged metrics."""
        n_metrics = len(self.metrics)
        if n_metrics == 0:
            return
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]
        
        for ax, (name, values) in zip(axes, self.metrics.items()):
            if isinstance(values[0], tuple):
                steps, vals = zip(*values)
                ax.plot(steps, vals)
                ax.set_xlabel('Step')
            else:
                ax.plot(values)
                ax.set_xlabel('Epoch')
            
            ax.set_ylabel(name)
            ax.set_title(name)
            ax.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, f'{self.experiment_name}_metrics.png')
        plt.savefig(plot_path)
        plt.close()


@dataclass
class ModelConfig:
    """Configuration for Transformer model."""
    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    d_ff: int = 2048
    max_seq_len: int = 100
    dropout: float = 0.1
    pos_encoding_type: str = 'rope'
    pad_idx: int = 0
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def create_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a comparison table for different experiments.
    
    Args:
        results: Dictionary of results from different experiments
        
    Returns:
        DataFrame with comparison metrics
    """
    data = []
    for exp_name, exp_results in results.items():
        row = {'Experiment': exp_name}
        for metric_name, metric_value in exp_results.items():
            if isinstance(metric_value, (int, float)):
                row[metric_name] = metric_value
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def plot_convergence_comparison(rope_losses: List[float], 
                               relative_losses: List[float],
                               save_path: str):
    """
    Plot convergence comparison between RoPE and Relative Position Bias.
    For the report analysis.
    """
    plt.figure(figsize=(10, 6))
    
    # Handle different length arrays
    rope_epochs = range(1, len(rope_losses) + 1)
    relative_epochs = range(1, len(relative_losses) + 1)
    
    # Plot each with its own epoch range
    plt.plot(rope_epochs, rope_losses, label='RoPE', marker='o', markersize=4)
    plt.plot(relative_epochs, relative_losses, label='Relative Position Bias', 
             marker='s', markersize=4)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Convergence Speed: RoPE vs Relative Position Bias', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add annotations for key points
    if rope_losses:
        min_rope = min(rope_losses)
        min_rope_epoch = rope_losses.index(min_rope) + 1
        plt.annotate(f'Min: {min_rope:.3f}', 
                    xy=(min_rope_epoch, min_rope),
                    xytext=(min_rope_epoch + 2, min_rope + 0.1),
                    arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7))
    
    if relative_losses:
        min_relative = min(relative_losses)
        min_relative_epoch = relative_losses.index(min_relative) + 1
        plt.annotate(f'Min: {min_relative:.3f}', 
                    xy=(min_relative_epoch, min_relative),
                    xytext=(min_relative_epoch + 2, min_relative + 0.1),
                    arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def analyze_decoding_outputs(results: Dict[str, Any], save_dir: str):
    """
    Analyze and visualize outputs from different decoding strategies.
    For the report analysis.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create BLEU comparison plot
    methods = list(results.keys())
    bleu_scores = [results[m]['bleu'] for m in methods]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(methods, bleu_scores, color=['blue', 'green', 'orange'])
    plt.ylabel('BLEU Score', fontsize=12)
    plt.title('BLEU Scores: Decoding Strategy Comparison', fontsize=14)
    
    # Add value labels on bars
    for bar, score in zip(bars, bleu_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'bleu_comparison.png'))
    plt.close()
    
    # Analyze text diversity (unique n-grams)
    diversity_stats = {}
    for method, data in results.items():
        predictions = data.get('predictions', [])
        if predictions:
            # Calculate unique bigrams and trigrams
            bigrams = set()
            trigrams = set()
            
            for pred in predictions[:100]:  # Analyze first 100
                tokens = pred.split()
                bigrams.update(zip(tokens[:-1], tokens[1:]))
                trigrams.update(zip(tokens[:-2], tokens[1:-1], tokens[2:]))
            
            diversity_stats[method] = {
                'unique_bigrams': len(bigrams),
                'unique_trigrams': len(trigrams),
                'avg_length': np.mean([len(p.split()) for p in predictions[:100]])
            }
    
    # Plot diversity metrics
    if diversity_stats:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['unique_bigrams', 'unique_trigrams', 'avg_length']
        titles = ['Unique Bigrams', 'Unique Trigrams', 'Average Length']
        
        for ax, metric, title in zip(axes, metrics, titles):
            values = [diversity_stats[m][metric] for m in methods]
            ax.bar(methods, values, color=['blue', 'green', 'orange'])
            ax.set_title(title)
            ax.set_ylabel('Count' if 'unique' in metric else 'Tokens')
            
            # Add value labels
            for i, v in enumerate(values):
                ax.text(i, v, f'{v:.1f}' if metric == 'avg_length' else str(v),
                       ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'diversity_analysis.png'))
        plt.close()
    
    return diversity_stats


def generate_latex_table(df: pd.DataFrame) -> str:
    """Convert DataFrame to LaTeX table format for the report."""
    return df.to_latex(index=False, float_format="%.2f")


def calculate_inference_time(model: nn.Module, 
                            input_tensor: torch.Tensor,
                            n_runs: int = 100) -> float:
    """Measure average inference time, handling both CPU and GPU."""
    model.eval()
    times = []
    
    # Handle CPU vs GPU timing
    if torch.cuda.is_available() and input_tensor.is_cuda:
        # GPU timing with CUDA events
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(input_tensor)
            
            # Actual timing
            for _ in range(n_runs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                _ = model(input_tensor)
                end.record()
                
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
    else:
        # CPU timing with time.time()
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(input_tensor)
            
            # Actual timing
            for _ in range(n_runs):
                start_time = time.time()
                _ = model(input_tensor)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
    
    return np.mean(times)