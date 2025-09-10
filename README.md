# Transformer from Scratch for Finnish-English Machine Translation

A complete implementation of the Transformer architecture for Finnish-to-English machine translation, comparing two positional encoding strategies (RoPE and Relative Position Bias) across three decoding methods (Greedy, Beam Search, and Top-k Sampling).

## Overview

This project implements the Transformer model architecture from scratch based on "Attention is All You Need" (Vaswani et al., 2017), with modern positional encoding techniques. The implementation includes:

- **Two Positional Encoding Methods**: 
  - Rotary Positional Embeddings (RoPE) - parameter-free geometric encoding
  - Relative Position Bias - learnable position-dependent attention biases
  
- **Three Decoding Strategies**:
  - Greedy Decoding - fast, deterministic
  - Beam Search - high-quality with length normalization and n-gram blocking
  - Top-k Sampling - diverse, creative outputs

- **Comprehensive Evaluation**: BLEU scores, convergence analysis, and qualitative assessment

## Key Results

| Positional Encoding | Decoding Strategy | BLEU Score |
|-------------------|------------------|------------|
| **RoPE** | **Beam Search** | **4.23** | 
| Relative Position Bias | Beam Search | 4.01 | 
| RoPE | Greedy | 3.46 | 
| Relative Position Bias | Greedy | 3.23 | 
| RoPE | Top-k Sampling | 2.79 | 
| Relative Position Bias | Top-k Sampling | 2.57 | 

**Key Findings**:
- RoPE converges 37.5% faster than Relative Position Bias
- Beam Search provides 31% BLEU improvement over Greedy baseline
- RoPE consistently outperforms Relative Position Bias across all decoding methods

## Architecture Specifications

- **Model**: Transformer encoder-decoder with 6 layers each
- **Dimensions**: d_model=512, n_heads=8, d_ff=2048
- **Parameters**: ~109.6M total
- **Vocabulary**: Finnish (64,786 tokens), English (31,522 tokens)
- **Maximum Sequence Length**: 100 tokens
- **Regularization**: Dropout (0.3), Label smoothing (0.1), Gradient clipping (0.5)

## Requirements

```bash
# Core dependencies
torch>=1.9.0
numpy>=1.21.0
matplotlib>=3.3.0
seaborn>=0.11.0
pandas>=1.3.0
sacrebleu>=2.0.0
tqdm>=4.62.0

# Optional for enhanced plotting
scikit-learn>=0.24.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ishaank1805/Transformer-from-Scratch-for-Finnish-English-Machine-Translation.git
```

2. Install dependencies:
```bash
pip install torch torchvision numpy matplotlib seaborn pandas sacrebleu tqdm
```

3. Prepare data files:
   - Download the EUbookshop Finnish-English parallel corpus
   - Place `EUbookshop.fi` and `EUbookshop.en` in the project root
   - The script will automatically create test splits

## Quick Start

### Complete Pipeline (Recommended)
Run the full experimental suite with one command:

```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

This script will:
1. Train both models (RoPE and Relative Position Bias)
2. Test all 6 configurations (2 encodings × 3 decoding methods)
3. Generate comprehensive evaluation results
4. Create report materials (plots, tables, statistics)

### Manual Training

Train individual models:

```bash
# Train with RoPE
python train.py \
    --src_file data/EUbookshop.fi \
    --tgt_file data/EUbookshop.en \
    --pos_encoding rope \
    --epochs 30 \
    --batch_size 32 \
    --d_model 512 \
    --n_heads 8 \
    --dropout 0.3 \
    --save_dir checkpoints \
    --use_amp

# Train with Relative Position Bias
python train.py \
    --src_file data/EUbookshop.fi \
    --tgt_file data/EUbookshop.en \
    --pos_encoding relative \
    --epochs 30 \
    --batch_size 32 \
    --d_model 512 \
    --n_heads 8 \
    --dropout 0.3 \
    --save_dir checkpoints \
    --use_amp
```

### Manual Testing

Test with different decoding strategies:

```bash
# Greedy decoding
python test.py \
    --checkpoint checkpoints/best_model_rope.pt \
    --vocab_path checkpoints/vocabs.pkl \
    --test_src test.fi \
    --test_tgt test.en \
    --decoding_method greedy \
    --output_dir results/rope_greedy

# Beam search
python test.py \
    --checkpoint checkpoints/best_model_rope.pt \
    --vocab_path checkpoints/vocabs.pkl \
    --test_src test.fi \
    --test_tgt test.en \
    --decoding_method beam \
    --beam_size 5 \
    --length_penalty 0.6 \
    --no_repeat_ngram_size 2 \
    --output_dir results/rope_beam

# Top-k sampling
python test.py \
    --checkpoint checkpoints/best_model_rope.pt \
    --vocab_path checkpoints/vocabs.pkl \
    --test_src test.fi \
    --test_tgt test.en \
    --decoding_method topk \
    --top_k 50 \
    --top_p 0.95 \
    --temperature 1.0 \
    --output_dir results/rope_topk
```

## Key Implementation Details

### Positional Encoding

**RoPE (Rotary Positional Embeddings)**:
- Parameter-free geometric encoding using rotation matrices
- Naturally captures relative position information
- Enables extrapolation to longer sequences
- Mathematical formulation: rotation by angle θᵢ = 10000^(-2i/d) for position m

**Relative Position Bias**:
- Learnable bias table for relative positions
- Adds position-dependent terms to attention scores
- 504 additional parameters for 8-head attention with max distance 32

### Decoding Strategies

**Beam Search**:
- Maintains top-5 hypotheses at each step
- Length penalty (α=0.6) prevents short translations
- N-gram blocking (n=2) reduces repetition
- Global optimization over sequence space

**Greedy Decoding**:
- Selects highest probability token at each step
- Fast inference but prone to repetition
- Deterministic output for reproducibility

**Top-k Sampling**:
- Samples from top-50 tokens with nucleus filtering (p=0.95)
- Introduces controlled randomness for diversity
- Temperature scaling (T=1.0) controls randomness

### Training Features

- **Mixed Precision Training**: Automatic mixed precision for memory efficiency
- **Early Stopping**: Patience-based stopping with validation monitoring
- **Gradient Clipping**: Prevents exploding gradients (max norm 0.5)
- **Advanced Regularization**: Dropout, label smoothing, weight decay
- **Noam Learning Rate Schedule**: Warmup + inverse sqrt decay




## Evaluation Metrics

The project uses BLEU (Bilingual Evaluation Understudy) scores for quantitative evaluation:
- **Corpus-level BLEU**: Overall translation quality across test set
- **Sentence-level BLEU**: Individual translation assessment
- **N-gram Precision**: 1-gram to 4-gram matching evaluation

Additional metrics available:
- Training/validation loss progression
- Convergence speed (epochs to optimal validation loss)
- Inference speed (tokens per second)
- Gradient stability analysis


**Poor BLEU Scores**:
- Check data preprocessing and vocabulary coverage
- Ensure proper train/validation/test splits
- Consider longer training with lower learning rate
- Verify beam search parameters (beam_size, length_penalty)



## Acknowledgments

- Original Transformer implementation based on "Attention is All You Need"
- RoPE implementation inspired by RoFormer paper
- EUbookshop corpus provided by OPUS
- Training optimizations adapted from modern NLP practices

## Model Links

https://drive.google.com/drive/folders/1mtH0umfcK61aoMeJFV1f2vSy2ANgWnoN?usp=sharing