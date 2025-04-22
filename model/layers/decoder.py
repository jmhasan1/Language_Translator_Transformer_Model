"""
Decoder layer and stack for the transformer model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.attention import MultiHeadAttention, PositionwiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Transformer decoder layer with self-attention, encoder-decoder attention, and feed-forward network.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Hidden dimension of the feed-forward network
            dropout: Dropout rate
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass through decoder layer.
        
        Args:
            x: Input tensor [batch_size, tgt_len, d_model]
            memory: Encoder output [batch_size, src_len, d_model]
            tgt_mask: Target sequence mask
            memory_mask: Memory mask
            tgt_key_padding_mask: Target padding mask
            memory_key_padding_mask: Memory padding mask
            
        Returns:
            Output tensor [batch_size, tgt_len, d_model]
            Self-attention weights
            Cross-attention weights
        """
        # Self-attention
        residual = x
        x, self_attn_weights = self.self_attn(x, x, x, tgt_mask, tgt_key_padding_mask)
        x = residual + self.dropout(x)
        x = self.norm1(x)
        
        # Cross-attention to encoder outputs
        residual = x
        x, cross_attn_weights = self.cross_attn(x, memory, memory, memory_mask, memory_key_padding_mask)
        x = residual + self.dropout(x)
        x = self.norm2(x)
        
        # Feed-forward
        residual = x
        x = self.feed_forward(x)
        x = residual + self.dropout(x)
        x = self.norm3(x)
        
        return x, self_attn_weights, cross_attn_weights


class Decoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Transformer decoder with multiple decoder layers.
        
        Args:
            num_layers: Number of decoder layers
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Hidden dimension of the feed-forward network
            dropout: Dropout rate
        """
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass through the decoder.
        
        Args:
            x: Input tensor [batch_size, tgt_len, d_model]
            memory: Encoder output [batch_size, src_len, d_model]
            tgt_mask: Target sequence mask
            memory_mask: Memory mask
            tgt_key_padding_mask: Target padding mask
            memory_key_padding_mask: Memory padding mask
            
        Returns:
            Output tensor [batch_size, tgt_len, d_model]
            List of self-attention weights
            List of cross-attention weights
        """
        self_attention_weights = []
        cross_attention_weights = []
        
        for layer in self.layers:
            x, self_attn, cross_attn = layer(
                x, memory, tgt_mask, memory_mask,
                tgt_key_padding_mask, memory_key_padding_mask
            )
            self_attention_weights.append(self_attn)
            cross_attention_weights.append(cross_attn)
        
        # Final layer normalization
        x = self.norm(x)
        
        return x, self_attention_weights, cross_attention_weights
