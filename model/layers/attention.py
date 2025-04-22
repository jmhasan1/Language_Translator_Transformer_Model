"""
Attention mechanisms for the transformer model.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Multi-head attention layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections for Q, K, V, and output
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize the projection matrices."""
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)
        
        nn.init.constant_(self.w_q.bias, 0)
        nn.init.constant_(self.w_k.bias, 0)
        nn.init.constant_(self.w_v.bias, 0)
        nn.init.constant_(self.w_o.bias, 0)
    
    def forward(self, q, k, v, mask=None, key_padding_mask=None):
        """
        Forward pass of multi-head attention.
        
        Args:
            q: Query tensor [batch_size, q_len, d_model]
            k: Key tensor [batch_size, k_len, d_model]
            v: Value tensor [batch_size, v_len, d_model]
            mask: Attention mask [q_len, k_len] or [batch_size, q_len, k_len]
            key_padding_mask: Key padding mask [batch_size, k_len]
            
        Returns:
            Output tensor [batch_size, q_len, d_model]
            Attention weights [batch_size, num_heads, q_len, k_len]
        """
        batch_size = q.size(0)
        q_len, k_len, v_len = q.size(1), k.size(1), v.size(1)
        
        # Linear projections and reshape for multi-head attention
        q = self.w_q(q).view(batch_size, q_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch, heads, q_len, d_k]
        k = self.w_k(k).view(batch_size, k_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch, heads, k_len, d_k]
        v = self.w_v(v).view(batch_size, v_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch, heads, v_len, d_k]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [batch, heads, q_len, k_len]
        
        # Apply masks
        if mask is not None:
            if mask.dim() == 2:  # [q_len, k_len]
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, k_len]
            elif mask.dim() == 3:  # [batch_size, q_len, k_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, q_len, k_len]
            
            scores = scores.masked_fill(mask == 0, -1e9)
        
        if key_padding_mask is not None:
            # [batch_size, 1, 1, k_len]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_padding_mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)  # [batch, heads, q_len, k_len]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)  # [batch, heads, q_len, d_k]
        
        # Reshape and project back to d_model
        context = context.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        output = self.w_o(context)
        
        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Position-wise feed-forward network.
        
        Args:
            d_model: Model dimension
            d_ff: Hidden dimension of the feed-forward network
            dropout: Dropout rate
        """
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize the weights."""
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)
        
        nn.init.constant_(self.w_1.bias, 0)
        nn.init.constant_(self.w_2.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
