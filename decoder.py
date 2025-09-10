"""
Transformer Decoder Implementation from Scratch
Advanced NLP Assignment 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple
from encoder import (
    MultiHeadAttention, 
    PositionwiseFeedForward, 
    LayerNorm,
    RotaryPositionalEmbedding,
    RelativePositionBias
)


class DecoderLayer(nn.Module):
    """Single Transformer Decoder Layer."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 pos_encoding_type: str = 'rope'):
        super().__init__()
        
        # Self-attention (masked)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Cross-attention (encoder-decoder attention)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.pos_encoding_type = pos_encoding_type
        
        # Calculate head dimension for RoPE
        head_dim = d_model // n_heads
        assert head_dim % 2 == 0, "RoPE requires even head dimension"
        
        # Initialize positional encoding based on type
        if pos_encoding_type == 'rope':
            self.rope = RotaryPositionalEmbedding(head_dim)  # Use head_dim, not d_model!
            self.rel_pos_bias = None
        elif pos_encoding_type == 'relative':
            self.rope = None
            self.rel_pos_bias = RelativePositionBias(n_heads)
        else:
            raise ValueError(f"Unknown positional encoding type: {pos_encoding_type}")
    
    def forward(self, 
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through decoder layer.
        
        Args:
            tgt: Target sequence tensor (batch_size, tgt_seq_len, d_model)
            memory: Encoder output (batch_size, src_seq_len, d_model)
            tgt_mask: Mask for target self-attention
            memory_mask: Mask for encoder-decoder attention
            
        Returns:
            Decoder layer output (batch_size, tgt_seq_len, d_model)
        """
        # Self-attention with causal mask
        if self.pos_encoding_type == 'rope':
            self_attn_output = self.self_attn(tgt, tgt, tgt, tgt_mask, rope=self.rope)
        else:  # relative position bias
            rel_bias = self.rel_pos_bias(tgt.size(1))
            self_attn_output = self.self_attn(tgt, tgt, tgt, tgt_mask, rel_pos_bias=rel_bias)
        
        tgt = self.norm1(tgt + self.dropout(self_attn_output))
        
        # Cross-attention with encoder output
        cross_attn_output = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout(cross_attn_output))
        
        # Feed-forward network
        ff_output = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout(ff_output))
        
        return tgt


class TransformerDecoder(nn.Module):
    """Complete Transformer Decoder Stack."""
    
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8,
                 n_layers: int = 6, d_ff: int = 2048, max_seq_len: int = 5000,
                 dropout: float = 0.1, pos_encoding_type: str = 'rope',
                 pad_idx: int = 0):
        super().__init__()
        
        assert pos_encoding_type in ['rope', 'relative'], \
            "Only 'rope' and 'relative' position encodings are supported"
        
        self.d_model = d_model
        self.pos_encoding_type = pos_encoding_type
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, pos_encoding_type)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for decoder self-attention.
        
        Returns a boolean mask where True = attend and False = mask out.
        Uses lower triangular matrix to ensure position i can only attend to positions <= i.
        """
        # True = keep, False = mask out
        # Lower triangular matrix (including diagonal)
        causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        return causal.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
    
    def create_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """Create padding mask for sequences.
        
        Returns a boolean mask where True = real token and False = padding.
        Shape: (batch_size, 1, 1, seq_len)
        """
        return (seq != self.pad_idx).unsqueeze(1).unsqueeze(2)
    
    def combine_masks(self, causal_mask: torch.Tensor, 
                     padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Combine causal and padding masks using boolean AND.
        
        Args:
            causal_mask: Causal attention mask with shape (1, 1, L, L)
            padding_mask: Padding mask with shape (B, 1, 1, S) that will be 
                         expanded to (B, 1, L, S) to match causal mask
        
        Returns:
            Combined mask with shape (B, 1, L, S) where True means "attend" and
            False means "mask out"
        """
        if padding_mask is None:
            return causal_mask
        
        # Expand padding mask to match causal mask dimensions
        seq_len = causal_mask.size(-1)
        padding_mask = padding_mask.expand(-1, -1, seq_len, -1)  # (B, 1, L, S)
        
        # Boolean AND: both must be True to allow attention
        return causal_mask & padding_mask
    
    def forward(self, 
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            tgt: Target sequence tensor (batch_size, tgt_seq_len)
            memory: Encoder output (batch_size, src_seq_len, d_model)
            tgt_mask: Optional target mask
            memory_mask: Optional memory mask (shape: B, 1, 1, S)
            
        Returns:
            Decoder output logits (batch_size, tgt_seq_len, vocab_size)
        """
        tgt_seq_len = tgt.size(1)
        device = tgt.device
        
        # Token embedding with scaling
        x = self.token_embedding(tgt) * self.scale
        
        x = self.dropout(x)
        
        # Create causal mask if not provided
        if tgt_mask is None:
            causal_mask = self.create_causal_mask(tgt_seq_len, device)
            padding_mask = self.create_padding_mask(tgt)
            tgt_mask = self.combine_masks(causal_mask, padding_mask)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        
        # Project to vocabulary size
        output = self.output_projection(x)
        
        return output
    
    def decode_step(self,
                   tgt: torch.Tensor,
                   memory: torch.Tensor,
                   memory_mask: Optional[torch.Tensor] = None,
                   cache: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Single decoding step for inference (used in beam search, greedy, etc.)
        
        Args:
            tgt: Target sequence so far (batch_size, current_seq_len)
            memory: Encoder output
            memory_mask: Encoder padding mask
            cache: Cache for faster decoding (optional)
            
        Returns:
            Tuple of (next_token_logits, updated_cache)
        """
        # Get decoder output
        output = self.forward(tgt, memory, memory_mask=memory_mask)
        
        # Return logits for last position only
        next_token_logits = output[:, -1, :]
        
        # For now, we're not implementing KV cache, but this is where it would go
        updated_cache = cache if cache else {}
        
        return next_token_logits, updated_cache


class EncoderDecoderTransformer(nn.Module):
    """Complete Encoder-Decoder Transformer Model."""
    
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_encoder_layers: int = 6,
                 n_decoder_layers: int = 6,
                 d_ff: int = 2048,
                 max_seq_len: int = 5000,
                 dropout: float = 0.1,
                 pos_encoding_type: str = 'rope',
                 pad_idx: int = 0):
        super().__init__()
        
        assert pos_encoding_type in ['rope', 'relative'], \
            "Only 'rope' and 'relative' position encodings are supported"
        
        # Import encoder here to avoid circular imports
        from encoder import TransformerEncoder
        
        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pos_encoding_type=pos_encoding_type
        )
        
        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_decoder_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pos_encoding_type=pos_encoding_type,
            pad_idx=pad_idx
        )
        
        self.pad_idx = pad_idx
        
    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through encoder-decoder.
        
        Args:
            src: Source sequence (batch_size, src_seq_len)
            tgt: Target sequence (batch_size, tgt_seq_len)
            src_mask: Source padding mask
            tgt_mask: Target mask (causal + padding)
            memory_mask: Memory mask for cross-attention (shape: B, 1, 1, S)
                        which broadcasts correctly to (B, H, L, S) in attention
            
        Returns:
            Output logits (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Create masks if not provided
        if src_mask is None:
            src_mask = self.encoder.create_padding_mask(src, self.pad_idx)
        
        if memory_mask is None:
            memory_mask = src_mask  # Reuse source mask for cross-attention
        
        # Encode source
        memory = self.encoder(src, src_mask)
        
        # Decode target
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        
        return output
    
    def encode(self, src: torch.Tensor, 
               src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode source sequence."""
        if src_mask is None:
            src_mask = self.encoder.create_padding_mask(src, self.pad_idx)
        return self.encoder(src, src_mask)
    
    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode target sequence given encoder output."""
        return self.decoder(tgt, memory, tgt_mask, memory_mask)