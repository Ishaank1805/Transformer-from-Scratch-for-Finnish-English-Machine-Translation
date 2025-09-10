"""
Testing script with different decoding strategies for Transformer model
Advanced NLP Assignment 1
Implements: Greedy, Beam Search, Top-k Sampling
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import json
import pickle
from typing import List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
from collections import defaultdict

from decoder import EncoderDecoderTransformer
from train import TranslationDataset, collate_fn, Vocabulary
from utils import sacre_bleu_from_pairs


@dataclass
class DecodingConfig:
    """Configuration for decoding strategies."""
    max_length: int = 100
    temperature: float = 1.0
    beam_size: int = 5
    top_k: int = 50
    top_p: float = 0.9
    length_penalty: float = 0.6
    no_repeat_ngram_size: int = 0
    early_stopping: bool = True


class DecodingStrategies:
    """Implementation of different decoding strategies."""
    
    def __init__(self, model: EncoderDecoderTransformer, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def greedy_decode(self, 
                     src: torch.Tensor,
                     max_length: int,
                     sos_idx: int,
                     eos_idx: int) -> torch.Tensor:
        """
        Greedy decoding - always select the token with highest probability.
        Based on the algorithm from the Hugging Face blog post.
        """
        self.model.eval()
        batch_size = src.size(0)
        
        # Encode source
        src_mask = self.model.encoder.create_padding_mask(src, self.model.pad_idx)
        memory = self.model.encode(src, src_mask)
        
        # Initialize with SOS token
        decoded = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=self.device)
        
        for _ in range(max_length - 1):
            # Get logits for next token
            output = self.model.decode(decoded, memory, memory_mask=src_mask)
            next_token_logits = output[:, -1, :]
            
            # Greedy selection: take argmax
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to decoded sequence
            decoded = torch.cat([decoded, next_token], dim=1)
            
            # Check if all sequences have produced EOS
            if (next_token == eos_idx).all():
                break
        
        return decoded
    
    def beam_search_decode(self,
                          src: torch.Tensor,
                          beam_size: int,
                          max_length: int,
                          sos_idx: int,
                          eos_idx: int,
                          length_penalty: float = 0.6,
                          no_repeat_ngram_size: int = 0) -> torch.Tensor:
        """
        Beam search decoding - maintain top-B sequences at each step.
        Implements the algorithm described in the Hugging Face blog.
        
        NOTE: For simplicity and clarity, we process each example in the batch
        independently. While less efficient than batched beam search, this approach
        is easier to understand and debug, which aligns with the assignment's
        focus on implementation clarity.
        """
        self.model.eval()
        batch_size = src.size(0)
        
        # For batch processing, we'll process each example separately
        # and collect results
        all_sequences = []
        
        for batch_idx in range(batch_size):
            # Get single source sequence
            src_single = src[batch_idx:batch_idx+1]
            
            # Encode source
            src_mask = self.model.encoder.create_padding_mask(src_single, self.model.pad_idx)
            memory = self.model.encode(src_single, src_mask)
            
            # Expand memory for beam size
            memory_expanded = memory.repeat(beam_size, 1, 1)
            src_mask_expanded = src_mask.repeat(beam_size, 1, 1, 1) if src_mask is not None else None
            
            # Initialize beams
            beam_scores = torch.zeros(beam_size, device=self.device)
            beam_scores[1:] = -1e9  # Only first beam is active initially
            
            # Start with SOS token
            beam_sequences = torch.full((beam_size, 1), sos_idx, 
                                       dtype=torch.long, device=self.device)
            
            # Track completed sequences
            completed_sequences = []
            completed_scores = []
            
            # Track which beams are active
            active_beam_mask = torch.ones(beam_size, dtype=torch.bool, device=self.device)
            
            for step in range(max_length - 1):
                # Only process active beams
                if not active_beam_mask.any():
                    break
                    
                # Get logits for next token
                output = self.model.decode(beam_sequences, memory_expanded, memory_mask=src_mask_expanded)
                next_token_logits = output[:, -1, :]
                
                # Apply n-gram blocking if specified
                if no_repeat_ngram_size > 0:
                    next_token_logits = self._block_ngrams(
                        beam_sequences, next_token_logits, no_repeat_ngram_size
                    )
                
                # Mask inactive beams
                next_token_logits[~active_beam_mask] = -float('inf')
                
                # Calculate scores (log probabilities)
                next_scores = F.log_softmax(next_token_logits, dim=-1)
                
                # Add to beam scores
                next_scores = beam_scores.unsqueeze(-1) + next_scores
                
                # Reshape for beam selection
                next_scores = next_scores.view(-1)
                
                # Select top beam_size sequences
                top_scores, top_indices = torch.topk(next_scores, beam_size)
                
                # Get beam and token indices
                beam_indices = top_indices // next_token_logits.size(-1)
                token_indices = top_indices % next_token_logits.size(-1)
                
                # Update beam sequences
                beam_sequences = torch.cat([
                    beam_sequences[beam_indices],
                    token_indices.unsqueeze(1)
                ], dim=1)
                
                # Update scores
                beam_scores = top_scores
                
                # Check for completed sequences (EOS token)
                new_active_mask = torch.ones_like(active_beam_mask)
                for i, token_id in enumerate(token_indices):
                    if token_id == eos_idx:
                        # Calculate length penalty for this beam
                        seq_len = beam_sequences[i].size(0)
                        lp = ((5 + seq_len) / 6) ** length_penalty
                        score_with_penalty = beam_scores[i].item() / lp
                        
                        completed_sequences.append(beam_sequences[i].clone())
                        completed_scores.append(score_with_penalty)
                        
                        # Mark beam as inactive
                        new_active_mask[i] = False
                        beam_scores[i] = -1e9
                
                active_beam_mask = active_beam_mask[beam_indices] & new_active_mask
                
                # Stop if we have enough completed sequences
                if len(completed_sequences) >= beam_size:
                    break
            
            # Select best sequence for this batch item
            if completed_sequences:
                best_idx = np.argmax(completed_scores)
                best_sequence = completed_sequences[best_idx]
            else:
                # Return best beam if no sequence completed
                best_sequence = beam_sequences[0]
            
            all_sequences.append(best_sequence)
        
        # Pad sequences to the same length before stacking
        max_seq_len = max(seq.size(0) for seq in all_sequences)
        padded_sequences = []
        
        for seq in all_sequences:
            if seq.size(0) < max_seq_len:
                # Use new_full to preserve device and dtype
                padding = seq.new_full((max_seq_len - seq.size(0),), eos_idx)
                padded_seq = torch.cat([seq, padding])
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)
        
        # Stack all sequences back into batch
        return torch.stack(padded_sequences, dim=0)
    
    def top_k_sampling(self,
                      src: torch.Tensor,
                      top_k: int,
                      max_length: int,
                      sos_idx: int,
                      eos_idx: int,
                      temperature: float = 1.0,
                      top_p: Optional[float] = None) -> torch.Tensor:
        """
        Top-k sampling with optional Top-p (nucleus) sampling.
        Implements the algorithm from the Hugging Face blog.
        """
        self.model.eval()
        batch_size = src.size(0)
        
        # Encode source
        src_mask = self.model.encoder.create_padding_mask(src, self.model.pad_idx)
        memory = self.model.encode(src, src_mask)
        
        # Initialize with SOS token
        decoded = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=self.device)
        
        for _ in range(max_length - 1):
            # Get logits for next token
            output = self.model.decode(decoded, memory, memory_mask=src_mask)
            next_token_logits = output[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                next_token_logits = self._top_k_filtering(next_token_logits, top_k)
            
            # Apply top-p (nucleus) filtering if specified
            if top_p is not None and top_p < 1.0:
                next_token_logits = self._top_p_filtering(next_token_logits, top_p)
            
            # Convert to probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to decoded sequence
            decoded = torch.cat([decoded, next_token], dim=1)
            
            # Check if all sequences have produced EOS
            if (next_token == eos_idx).all():
                break
        
        return decoded
    
    def _top_k_filtering(self, logits: torch.Tensor, k: int) -> torch.Tensor:
           """
           Filter logits to keep only top k tokens.
           Sets all other logits to -inf.
           """
           if k == 0:
             return logits
     
           # Clamp k to vocabulary size to avoid errors
           k = min(k, logits.size(-1))
    
           values, indices = torch.topk(logits, k)
           min_values = values[:, -1].unsqueeze(-1)
    
           # Set all logits below the k-th highest to -inf
           return torch.where(logits < min_values, 
                      torch.full_like(logits, float('-inf')), 
                      logits)
    
    def _top_p_filtering(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """
        Filter logits using nucleus (top-p) sampling.
        Keeps the smallest set of tokens whose cumulative probability >= p.
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Find cutoff: first position where cumulative prob exceeds p
        sorted_indices_to_remove = cumulative_probs > p
        
        # Shift the indices to the right to keep the first token above threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        
        # Create mask in original order
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        
        # Set filtered positions to -inf
        return logits.masked_fill(indices_to_remove, float('-inf'))
    
    def _block_ngrams(self, sequences: torch.Tensor, logits: torch.Tensor, 
                     ngram_size: int) -> torch.Tensor:
        """
        Block repeated n-grams by setting their probability to -inf.
        Note: This implementation uses Python loops which may be slow for large vocabularies.
        For production use, consider a vectorized implementation.
        """
        batch_size, seq_len = sequences.shape
        
        # Handle edge cases
        if ngram_size <= 0 or seq_len < ngram_size:
            return logits
        
        for batch_idx in range(batch_size):
            # Get the current sequence as a list
            seq = sequences[batch_idx].tolist()
            
            # Find all n-grams of size ngram_size
            if len(seq) >= ngram_size:
                ngrams = set()
                for i in range(len(seq) - ngram_size + 1):
                    ngram = tuple(seq[i:i + ngram_size])
                    ngrams.add(ngram)
                
                # Check which tokens would create repeated n-grams
                if len(seq) >= ngram_size - 1:
                    prefix = tuple(seq[-(ngram_size - 1):])
                    for token_id in range(logits.size(-1)):
                        candidate_ngram = prefix + (token_id,)
                        if candidate_ngram in ngrams:
                            logits[batch_idx, token_id] = float('-inf')
        
        return logits


class TranslationEvaluator:
    """Evaluator for translation quality."""
    
    def __init__(self, model: EncoderDecoderTransformer, 
                 test_loader: DataLoader,
                 src_vocab: Vocabulary,
                 tgt_vocab: Vocabulary,
                 device: torch.device):
        self.model = model
        self.test_loader = test_loader
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.decoder = DecodingStrategies(model, device)
    
    def decode_sequence(self, token_ids: torch.Tensor, vocab: Vocabulary) -> str:
        """Convert token IDs back to text."""
        # Ensure token_ids is 1D
        if token_ids.dim() == 0:
            token_ids = token_ids.unsqueeze(0)
        elif token_ids.dim() > 1:
            token_ids = token_ids.squeeze()
        
        # Convert to list once for efficiency
        if hasattr(token_ids, 'tolist'):
            ids = token_ids.tolist()
        else:
            ids = [token_ids.item()] if hasattr(token_ids, 'item') else [token_ids]
        
        tokens = []
        for token_id in ids:
            if token_id == vocab.stoi["<EOS>"]:
                break
            if token_id not in [vocab.stoi["<PAD>"], vocab.stoi["<SOS>"]]:
                tokens.append(vocab.itos.get(token_id, "<UNK>"))
        return " ".join(tokens)
    
    def evaluate(self, decoding_method: str, config: DecodingConfig) -> dict:
        """
        Evaluate model with specified decoding strategy.
        Returns BLEU scores and example translations.
        """
        self.model.eval()
        
        predictions = []
        references = []
        examples = []
        
        sos_idx = self.tgt_vocab.stoi["<SOS>"]
        eos_idx = self.tgt_vocab.stoi["<EOS>"]
        
        print(f"\nEvaluating with {decoding_method} decoding...")
        
        with torch.no_grad():
            for batch_idx, (src, tgt) in enumerate(tqdm(self.test_loader)):
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                
                # Decode based on method
                if decoding_method == 'greedy':
                    output = self.decoder.greedy_decode(
                        src, config.max_length, sos_idx, eos_idx
                    )
                elif decoding_method == 'beam':
                    output = self.decoder.beam_search_decode(
                        src, config.beam_size, config.max_length, 
                        sos_idx, eos_idx, config.length_penalty,
                        config.no_repeat_ngram_size
                    )
                elif decoding_method == 'topk':
                    output = self.decoder.top_k_sampling(
                        src, config.top_k, config.max_length,
                        sos_idx, eos_idx, config.temperature, config.top_p
                    )
                else:
                    raise ValueError(f"Unknown decoding method: {decoding_method}")
                
                # Process each sequence in batch
                for i in range(src.size(0)):
                    # Get source, prediction, and reference
                    src_text = self.decode_sequence(src[i], self.src_vocab)
                    
                    # Extract prediction for this batch item
                    if output.dim() > 1:
                        pred_sequence = output[i]
                    else:
                        # Handle single sequence case
                        pred_sequence = output
                    
                    pred_text = self.decode_sequence(pred_sequence, self.tgt_vocab)
                    ref_text = self.decode_sequence(tgt[i], self.tgt_vocab)
                    
                    predictions.append(pred_text)
                    references.append([ref_text])  # BLEU expects list of references
                    
                    # Store examples (first 5 batches)
                    if batch_idx < 5 and i < 2:
                        examples.append({
                            'source': src_text,
                            'reference': ref_text,
                            'prediction': pred_text
                        })
        
        # Calculate BLEU score using the fixed sacrebleu function
        bleu_score = sacre_bleu_from_pairs(predictions, references)
        
        return {
            'bleu': bleu_score,
            'predictions': predictions[:100],  # Store first 100 for analysis
            'references': references[:100],
            'examples': examples,
            'method': decoding_method
        }


def main():
    parser = argparse.ArgumentParser(description='Test Transformer with different decoding strategies')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--vocab_path', type=str, default='checkpoints/vocabs.pkl',
                       help='Path to vocabulary file')
    
    # Data arguments
    parser.add_argument('--test_src', type=str, default='test.fi',
                       help='Test source file')
    parser.add_argument('--test_tgt', type=str, default='test.en',
                       help='Test target file')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for testing')
    
    # Decoding arguments
    parser.add_argument('--decoding_method', type=str, default='all',
                       choices=['greedy', 'beam', 'topk', 'all'],
                       help='Decoding method to use')
    parser.add_argument('--max_length', type=int, default=100,
                       help='Maximum sequence length')
    parser.add_argument('--beam_size', type=int, default=5,
                       help='Beam size for beam search')
    parser.add_argument('--top_k', type=int, default=50,
                       help='K for top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.95,
                       help='P for nucleus sampling')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for sampling')
    parser.add_argument('--length_penalty', type=float, default=0.6,
                       help='Length penalty for beam search')
    parser.add_argument('--no_repeat_ngram_size', type=int, default=2,
                       help='Size of n-grams to block in beam search')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_args = checkpoint['args']
    
    # Load vocabularies
    with open(args.vocab_path, 'rb') as f:
        vocabs = pickle.load(f)
        src_vocab = vocabs['src_vocab']
        tgt_vocab = vocabs['tgt_vocab']
    
    print(f"Loaded vocabularies - Source: {len(src_vocab)}, Target: {len(tgt_vocab)}")
    
    # Create test dataset
    test_dataset = TranslationDataset(
        args.test_src, args.test_tgt,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_length=args.max_length
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = EncoderDecoderTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=model_args.d_model,
        n_heads=model_args.n_heads,
        n_encoder_layers=model_args.n_layers,
        n_decoder_layers=model_args.n_layers,
        d_ff=model_args.d_ff,
        max_seq_len=model_args.max_seq_len,
        dropout=0.0,  # No dropout during testing
        pos_encoding_type=model_args.pos_encoding,
        pad_idx=0
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully. Testing on {device}")
    
    # Create evaluator
    evaluator = TranslationEvaluator(model, test_loader, src_vocab, tgt_vocab, device)
    
    # Create decoding config
    config = DecodingConfig(
        max_length=args.max_length,
        temperature=args.temperature,
        beam_size=args.beam_size,
        top_k=args.top_k,
        top_p=args.top_p,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size
    )
    
    # Determine which methods to evaluate
    if args.decoding_method == 'all':
        methods = ['greedy', 'beam', 'topk']
    else:
        methods = [args.decoding_method]
    
    # Evaluate each method
    results = {}
    for method in methods:
        result = evaluator.evaluate(method, config)
        results[method] = result
        
        print(f"\n{method.upper()} Decoding Results:")
        print(f"BLEU Score: {result['bleu']:.2f}")
        
        # Print examples
        print(f"\nExample Translations ({method}):")
        print("-" * 80)
        for i, example in enumerate(result['examples'][:3]):
            print(f"Example {i+1}:")
            print(f"Source:     {example['source']}")
            print(f"Reference:  {example['reference']}")
            print(f"Prediction: {example['prediction']}")
            print("-" * 80)
    
    # Save results
    results_file = os.path.join(args.output_dir, 
                               f"results_{model_args.pos_encoding}.json")
    
    # Prepare results for JSON serialization
    json_results = {}
    for method, result in results.items():
        json_results[method] = {
            'bleu': result['bleu'],
            'examples': result['examples'],
            'sample_predictions': result['predictions'][:20],
            'sample_references': [ref[0] for ref in result['references'][:20]]
        }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Print summary comparison if all methods were evaluated
    if len(methods) > 1:
        print("\n" + "="*50)
        print("SUMMARY - BLEU Scores Comparison:")
        print("="*50)
        for method in methods:
            print(f"{method.upper():15} : {results[method]['bleu']:.2f}")
        print("="*50)


if __name__ == '__main__':
    main()