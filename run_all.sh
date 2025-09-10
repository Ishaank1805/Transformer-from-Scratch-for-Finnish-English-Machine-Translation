#!/bin/bash

# Advanced NLP Assignment 1: Complete Pipeline
# This script runs all experiments required for the assignment

set -e  # Exit on error

echo "=================================================="
echo "Advanced NLP Assignment 1: Transformer from Scratch"
echo "=================================================="

# Configuration
DATA_DIR="data"
CHECKPOINT_DIR="checkpoints"
RESULTS_DIR="results"
PLOTS_DIR="plots"
REPORT_DIR="report_materials"

# Training hyperparameters
EPOCHS=30
BATCH_SIZE=32
LEARNING_RATE=0.0001
D_MODEL=512
N_HEADS=8
N_LAYERS=6
D_FF=2048
MAX_SEQ_LEN=100
DROPOUT=0.3  # UPDATED: Increased from 0.1 for regularization
CLIP_GRAD=0.5  # NEW: Gradient clipping
EARLY_STOPPING_PATIENCE=5  # NEW: Early stopping patience
MIN_DELTA=0.01  # NEW: Minimum improvement delta

# Create directories
echo "Creating directories..."
mkdir -p $DATA_DIR $CHECKPOINT_DIR $RESULTS_DIR $PLOTS_DIR $REPORT_DIR

# Check if data files exist
if [ ! -f "EUbookshop.fi" ] || [ ! -f "EUbookshop.en" ]; then
    echo "Error: Data files EUbookshop.fi and EUbookshop.en not found!"
    echo "Please ensure the parallel corpus files are in the current directory."
    exit 1
fi

# Move data files to data directory if not already there
if [ ! -f "$DATA_DIR/EUbookshop.fi" ]; then
    echo "Moving data files to $DATA_DIR directory..."
    cp EUbookshop.fi $DATA_DIR/
    cp EUbookshop.en $DATA_DIR/
fi

echo ""
echo "=================================================="
echo "PHASE 1: Training Transformers"
echo "=================================================="

# 1. Train with RoPE
echo ""
echo "Training Transformer with RoPE..."
echo "--------------------------------------------------"
python train.py \
    --src_file $DATA_DIR/EUbookshop.fi \
    --tgt_file $DATA_DIR/EUbookshop.en \
    --pos_encoding rope \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_layers $N_LAYERS \
    --d_ff $D_FF \
    --max_seq_len $MAX_SEQ_LEN \
    --dropout $DROPOUT \
    --clip_grad $CLIP_GRAD \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --min_delta $MIN_DELTA \
    --save_dir $CHECKPOINT_DIR \
    --plot_interval 5 \
    --use_amp

# 2. Train with Relative Position Bias
echo ""
echo "Training Transformer with Relative Position Bias..."
echo "--------------------------------------------------"
python train.py \
    --src_file $DATA_DIR/EUbookshop.fi \
    --tgt_file $DATA_DIR/EUbookshop.en \
    --pos_encoding relative \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_layers $N_LAYERS \
    --d_ff $D_FF \
    --max_seq_len $MAX_SEQ_LEN \
    --dropout $DROPOUT \
    --clip_grad $CLIP_GRAD \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --min_delta $MIN_DELTA \
    --save_dir $CHECKPOINT_DIR \
    --plot_interval 5 \
    --use_amp

echo ""
echo "=================================================="
echo "PHASE 2: Testing with Different Decoding Strategies"
echo "=================================================="

# Test RoPE model with all decoding strategies
echo ""
echo "Testing RoPE model..."
echo "--------------------------------------------------"

# 2.1 Greedy Decoding
echo "Testing RoPE with Greedy Decoding..."
python test.py \
    --checkpoint $CHECKPOINT_DIR/best_model_rope.pt \
    --vocab_path $CHECKPOINT_DIR/vocabs.pkl \
    --test_src test.fi \
    --test_tgt test.en \
    --decoding_method greedy \
    --batch_size 32 \
    --max_length 100 \
    --output_dir $RESULTS_DIR/rope_greedy

# 2.2 Beam Search
echo "Testing RoPE with Beam Search..."
python test.py \
    --checkpoint $CHECKPOINT_DIR/best_model_rope.pt \
    --vocab_path $CHECKPOINT_DIR/vocabs.pkl \
    --test_src test.fi \
    --test_tgt test.en \
    --decoding_method beam \
    --beam_size 5 \
    --length_penalty 0.6 \
    --no_repeat_ngram_size 2 \
    --batch_size 32 \
    --max_length 100 \
    --output_dir $RESULTS_DIR/rope_beam

# 2.3 Top-k Sampling
echo "Testing RoPE with Top-k Sampling..."
python test.py \
    --checkpoint $CHECKPOINT_DIR/best_model_rope.pt \
    --vocab_path $CHECKPOINT_DIR/vocabs.pkl \
    --test_src test.fi \
    --test_tgt test.en \
    --decoding_method topk \
    --top_k 50 \
    --top_p 0.95 \
    --temperature 1.0 \
    --batch_size 32 \
    --max_length 100 \
    --output_dir $RESULTS_DIR/rope_topk

# Test Relative Position Bias model with all decoding strategies
echo ""
echo "Testing Relative Position Bias model..."
echo "--------------------------------------------------"

# 2.4 Greedy Decoding
echo "Testing Relative with Greedy Decoding..."
python test.py \
    --checkpoint $CHECKPOINT_DIR/best_model_relative.pt \
    --vocab_path $CHECKPOINT_DIR/vocabs.pkl \
    --test_src test.fi \
    --test_tgt test.en \
    --decoding_method greedy \
    --batch_size 32 \
    --max_length 100 \
    --output_dir $RESULTS_DIR/relative_greedy

# 2.5 Beam Search
echo "Testing Relative with Beam Search..."
python test.py \
    --checkpoint $CHECKPOINT_DIR/best_model_relative.pt \
    --vocab_path $CHECKPOINT_DIR/vocabs.pkl \
    --test_src test.fi \
    --test_tgt test.en \
    --decoding_method beam \
    --beam_size 5 \
    --length_penalty 0.6 \
    --no_repeat_ngram_size 2 \
    --batch_size 32 \
    --max_length 100 \
    --output_dir $RESULTS_DIR/relative_beam

# 2.6 Top-k Sampling
echo "Testing Relative with Top-k Sampling..."
python test.py \
    --checkpoint $CHECKPOINT_DIR/best_model_relative.pt \
    --vocab_path $CHECKPOINT_DIR/vocabs.pkl \
    --test_src test.fi \
    --test_tgt test.en \
    --decoding_method topk \
    --top_k 50 \
    --top_p 0.95 \
    --temperature 1.0 \
    --batch_size 32 \
    --max_length 100 \
    --output_dir $RESULTS_DIR/relative_topk

echo ""
echo "=================================================="
echo "PHASE 3: Comprehensive Evaluation"
echo "=================================================="

# Run comprehensive evaluation with all methods at once
echo "Running comprehensive evaluation..."
python test.py \
    --checkpoint $CHECKPOINT_DIR/best_model_rope.pt \
    --vocab_path $CHECKPOINT_DIR/vocabs.pkl \
    --test_src test.fi \
    --test_tgt test.en \
    --decoding_method all \
    --beam_size 5 \
    --top_k 50 \
    --top_p 0.95 \
    --batch_size 32 \
    --max_length 100 \
    --output_dir $RESULTS_DIR/rope_all

python test.py \
    --checkpoint $CHECKPOINT_DIR/best_model_relative.pt \
    --vocab_path $CHECKPOINT_DIR/vocabs.pkl \
    --test_src test.fi \
    --test_tgt test.en \
    --decoding_method all \
    --beam_size 5 \
    --top_k 50 \
    --top_p 0.95 \
    --batch_size 32 \
    --max_length 100 \
    --output_dir $RESULTS_DIR/relative_all

echo ""
echo "=================================================="
echo "PHASE 4: Generate Report Materials"
echo "=================================================="

# Create analysis script for report generation
cat > generate_report_materials.py << 'EOF'
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import numpy as np
from utils import plot_convergence_comparison, analyze_decoding_outputs, create_comparison_table

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("Generating report materials...")

# Load training histories
with open('checkpoints/best_model_rope.pt', 'rb') as f:
    import torch
    rope_checkpoint = torch.load(f, map_location='cpu')
    rope_train_losses = rope_checkpoint['train_losses']
    rope_val_losses = rope_checkpoint['val_losses']

with open('checkpoints/best_model_relative.pt', 'rb') as f:
    relative_checkpoint = torch.load(f, map_location='cpu')
    relative_train_losses = relative_checkpoint['train_losses']
    relative_val_losses = relative_checkpoint['val_losses']

# 1. Convergence comparison plot
print("Creating convergence comparison plot...")
plot_convergence_comparison(rope_train_losses, relative_train_losses, 
                           'report_materials/convergence_comparison.png')

# 2. Create BLEU score comparison table
print("Creating BLEU score comparison table...")
results_summary = {
    'Configuration': [],
    'BLEU Score': [],
    'Positional Encoding': [],
    'Decoding Strategy': []
}

# Load all results
result_files = [
    ('rope_all/results_rope.json', 'RoPE'),
    ('relative_all/results_relative.json', 'Relative')
]

for file_path, encoding in result_files:
    full_path = f'results/{file_path}'
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            data = json.load(f)
            for method in ['greedy', 'beam', 'topk']:
                if method in data:
                    results_summary['Configuration'].append(f'{encoding}_{method}')
                    results_summary['BLEU Score'].append(data[method]['bleu'])
                    results_summary['Positional Encoding'].append(encoding)
                    results_summary['Decoding Strategy'].append(method.capitalize())

df_results = pd.DataFrame(results_summary)

# Save as CSV
df_results.to_csv('report_materials/bleu_scores_table.csv', index=False)

# Create visualization
print("Creating BLEU score visualization...")
fig, ax = plt.subplots(figsize=(12, 6))

# Group by encoding and strategy
rope_data = df_results[df_results['Positional Encoding'] == 'RoPE']
relative_data = df_results[df_results['Positional Encoding'] == 'Relative']

x = np.arange(3)  # for 3 decoding strategies
width = 0.35

strategies = ['Greedy', 'Beam', 'Topk']
rope_scores = [rope_data[rope_data['Decoding Strategy'] == s]['BLEU Score'].values[0] 
               if len(rope_data[rope_data['Decoding Strategy'] == s]) > 0 else 0 
               for s in strategies]
relative_scores = [relative_data[relative_data['Decoding Strategy'] == s]['BLEU Score'].values[0] 
                  if len(relative_data[relative_data['Decoding Strategy'] == s]) > 0 else 0 
                  for s in strategies]

bars1 = ax.bar(x - width/2, rope_scores, width, label='RoPE')
bars2 = ax.bar(x + width/2, relative_scores, width, label='Relative Position Bias')

ax.set_xlabel('Decoding Strategy', fontsize=12)
ax.set_ylabel('BLEU Score', fontsize=12)
ax.set_title('BLEU Scores: Positional Encoding × Decoding Strategy', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(strategies)
ax.legend()

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('report_materials/bleu_comparison_chart.png', dpi=150)
plt.close()

# 3. Generate LaTeX table for report
print("Generating LaTeX table...")
latex_table = df_results.to_latex(index=False, float_format="%.2f", 
                                  caption="BLEU scores for different configurations",
                                  label="tab:bleu_scores")

with open('report_materials/bleu_table.tex', 'w') as f:
    f.write(latex_table)

# 4. Training statistics summary
print("Creating training statistics summary...")
stats_summary = {
    'Model': ['RoPE', 'Relative Position Bias'],
    'Final Train Loss': [rope_train_losses[-1] if rope_train_losses else 0, 
                         relative_train_losses[-1] if relative_train_losses else 0],
    'Final Val Loss': [rope_val_losses[-1] if rope_val_losses else 0,
                       relative_val_losses[-1] if relative_val_losses else 0],
    'Best Val Loss': [min(rope_val_losses) if rope_val_losses else 0,
                      min(relative_val_losses) if relative_val_losses else 0],
    'Convergence Epoch': [rope_val_losses.index(min(rope_val_losses)) + 1 if rope_val_losses else 0,
                          relative_val_losses.index(min(relative_val_losses)) + 1 if relative_val_losses else 0]
}

df_stats = pd.DataFrame(stats_summary)
df_stats.to_csv('report_materials/training_statistics.csv', index=False)

print("\nReport materials generated successfully!")
print("Files created in report_materials/:")
print("  - convergence_comparison.png")
print("  - bleu_scores_table.csv")
print("  - bleu_comparison_chart.png")
print("  - bleu_table.tex")
print("  - training_statistics.csv")
EOF

python generate_report_materials.py

echo ""
echo "=================================================="
echo "PHASE 5: Final Summary"
echo "=================================================="

# Create final summary
cat > create_summary.py << 'EOF'
import json
import os
import pandas as pd

print("\n" + "="*60)
print("ASSIGNMENT COMPLETION SUMMARY")
print("="*60)

# Check what files were created
required_files = {
    "Training Checkpoints": [
        "checkpoints/best_model_rope.pt",
        "checkpoints/best_model_relative.pt"
    ],
    "Result Files": [
        "results/rope_all/results_rope.json",
        "results/relative_all/results_relative.json"
    ],
    "Report Materials": [
        "report_materials/convergence_comparison.png",
        "report_materials/bleu_scores_table.csv",
        "report_materials/bleu_comparison_chart.png"
    ]
}

print("\nFile Status:")
print("-" * 40)
for category, files in required_files.items():
    print(f"\n{category}:")
    for file in files:
        status = "✓" if os.path.exists(file) else "✗"
        print(f"  {status} {file}")

# Load and display final BLEU scores
print("\n" + "="*60)
print("FINAL BLEU SCORES")
print("="*60)

if os.path.exists('report_materials/bleu_scores_table.csv'):
    df = pd.read_csv('report_materials/bleu_scores_table.csv')
    print("\n" + df.to_string(index=False))

print("\n" + "="*60)
print("Assignment completed successfully!")
print("All required experiments have been run.")
print("Report materials are in: report_materials/")
print("="*60)
EOF

python create_summary.py

# Cleanup temporary Python scripts
rm -f generate_report_materials.py create_summary.py

echo ""
echo "=================================================="
echo "All experiments completed!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Check the 'report_materials' directory for plots and tables"
echo "2. Review the BLEU scores in 'results' directory"
echo "3. Use the generated materials for your report"
echo ""
echo "To retrain with different hyperparameters, modify the variables"
echo "at the top of this script and run again."
echo ""