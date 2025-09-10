"""
Transformer Encoder Implementation from Scratch
Advanced NLP Assignment 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism implemented from scratch."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # RoPE requires even head dimension
        assert self.d_k % 2 == 0, "RoPE requires even head dimension"
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                rope: Optional['RotaryPositionalEmbedding'] = None,
                rel_pos_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Extract dimensions properly
        batch_size, query_len, _ = query.size()
        _, key_len, _ = key.size()
        
        # Linear transformations and reshape to (batch, heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, query_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, key_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, key_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply RoPE if provided (only for self-attention where query_len == key_len)
        if rope is not None and query_len == key_len:
            Q = rope.apply_rotary_pos_emb(Q)
            K = rope.apply_rotary_pos_emb(K)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Add relative position bias if provided
        if rel_pos_bias is not None:
            scores = scores + rel_pos_bias
        
        # Apply mask if provided
        if mask is not None:
            # mask is boolean: True = attend, False = mask out
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.d_model
        )
        output = self.W_o(context)
        
        return output


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class LayerNorm(nn.Module):
    """Layer Normalization implemented from scratch."""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)  # Use population variance
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) implementation."""
    
    def __init__(self, dim: int, max_seq_len: int = 5000, base: int = 10000):
        super().__init__()
        self.dim = dim  # This should be head_dim (d_k), not d_model
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute the frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin for max sequence length
        self._precompute_theta(max_seq_len)
    
    def _precompute_theta(self, seq_len: int):
        position = torch.arange(seq_len).unsqueeze(1)
        freqs = position * self.inv_freq
        # Don't concatenate - keep the D//2 dimension
        self.register_buffer('cos_cached', freqs.cos())  # Shape: (seq_len, D//2)
        self.register_buffer('sin_cached', freqs.sin())  # Shape: (seq_len, D//2)
    
    def apply_rotary_pos_emb(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embedding to input tensor."""
        # x: (B, H, L, D), D even
        B, H, L, D = x.shape
        
        if L > self.max_seq_len:
            self._precompute_theta(L)
            # Move to correct device after recomputing
            self.cos_cached = self.cos_cached.to(x.device)
            self.sin_cached = self.sin_cached.to(x.device)
        
        # Get cached cos and sin values
        cos = self.cos_cached[:L].unsqueeze(0).unsqueeze(0)  # (1, 1, L, D//2)
        sin = self.sin_cached[:L].unsqueeze(0).unsqueeze(0)  # (1, 1, L, D//2)
        
        # Split x into first half and second half
        x1, x2 = x[..., :D//2], x[..., D//2:]
        
        # Apply rotation using the "rotate half" method
        # (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        
        return torch.cat([x1_rot, x2_rot], dim=-1)


class RelativePositionBias(nn.Module):
    """Relative Position Bias implementation."""
    
    def __init__(self, n_heads: int, max_relative_position: int = 32):
        super().__init__()
        self.n_heads = n_heads
        self.max_relative_position = max_relative_position
        
        # Create relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * max_relative_position - 1, n_heads))
        )
        nn.init.xavier_uniform_(self.relative_position_bias_table)
        
    def forward(self, seq_len: int) -> torch.Tensor:
        """Compute relative position bias for given sequence length."""
        # Create position indices on the same device as the bias table
        positions = torch.arange(seq_len, device=self.relative_position_bias_table.device)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        
        # Clip to max relative position
        relative_positions = relative_positions.clamp(
            -self.max_relative_position + 1,
            self.max_relative_position - 1
        )
        
        # Shift to positive indices
        relative_positions += self.max_relative_position - 1
        
        # Get bias values
        bias = self.relative_position_bias_table[relative_positions]
        
        # Reshape to (1, n_heads, seq_len, seq_len)
        return bias.permute(2, 0, 1).unsqueeze(0)


class EncoderLayer(nn.Module):
    """Single Transformer Encoder Layer."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 pos_encoding_type: str = 'rope'):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
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
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        if self.pos_encoding_type == 'rope':
            attn_output = self.self_attn(x, x, x, mask, rope=self.rope)
        else:  # relative position bias
            rel_bias = self.rel_pos_bias(x.size(1))
            attn_output = self.self_attn(x, x, x, mask, rel_pos_bias=rel_bias)
        
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """Complete Transformer Encoder Stack."""
    
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8,
                 n_layers: int = 6, d_ff: int = 2048, max_seq_len: int = 5000,
                 dropout: float = 0.1, pos_encoding_type: str = 'rope'):
        super().__init__()
        
        assert pos_encoding_type in ['rope', 'relative'], \
            "Only 'rope' and 'relative' position encodings are supported"
        
        self.d_model = d_model
        self.pos_encoding_type = pos_encoding_type
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, pos_encoding_type)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            src: Source sequence tensor of shape (batch_size, seq_len)
            mask: Optional mask tensor
            
        Returns:
            Encoded representation of shape (batch_size, seq_len, d_model)
        """
        # Token embedding with scaling
        x = self.token_embedding(src) * self.scale
        
        x = self.dropout(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return x
    
    def create_padding_mask(self, src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """Create padding mask for source sequence."""
        # Shape: (batch_size, 1, 1, seq_len)
        # Boolean mask: True = attend, False = mask out
        mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        return mask