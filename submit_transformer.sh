#!/bin/bash
#SBATCH --job-name=transformer_train
#SBATCH --output=transformer_%j.out
#SBATCH --error=transformer_%j.err
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -A research
#SBATCH --qos=medium

# Load bash configuration
source ~/.bashrc

# Activate conda environment
conda activate pyg

# Make sure the script is executable
chmod +x run_all.sh

# Run the complete pipeline
./run_all.sh