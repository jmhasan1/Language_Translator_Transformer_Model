"""
Encoder layer and stack for the transformer model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.attention import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Transformer encoder layer with self-attention and feed-forward network.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Hidden dimension of the feed-forward network
            dropout: Dropout rate
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None, src_key_padding_mask=None):
        """
        Forward pass through encoder layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask
            src_key_padding_mask: Source padding mask
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
            Attention weights
        """
        # Self-attention with residual connection and layer normalization
        residual = x
        x, attn_weights = self.self_attn(x, x, x, mask, src_key_padding_mask)
        x = residual + self.dropout(x)
        x = self.norm1(x)
        
        # Feed-forward with residual connection and layer normalization
        residual = x
        x = self.feed_forward(x)
        x = residual + self.dropout(x)
        x = self.norm2(x)
        
        return x, attn_weights


class Encoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Transformer encoder with multiple encoder layers.
        
        Args:
            num_layers: Number of encoder layers
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Hidden dimension of the feed-forward network
            dropout: Dropout rate
        """
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None, src_key_padding_mask=None):
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask
            src_key_padding_mask: Source padding mask
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
            List of attention weights from each layer
        """
        attention_weights = []
        
        for layer in self.layers:
            x, attn_weights = layer(x, mask, src_key_padding_mask)
            attention_weights.append(attn_weights)
        
        # Final layer normalization
        x = self.norm(x)
        
        return x, attention_weights
