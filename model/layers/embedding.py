"""
Embedding layers for the transformer model.
"""

import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        """
        Token embedding layer.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
        # Initialize embeddings
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of token indices [batch_size, seq_len]
            
        Returns:
            Embedded tensor [batch_size, seq_len, d_model]
        """
        # Multiply by sqrt(d_model) to scale embeddings as in the paper
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        """
        Positional encoding layer.
        
        Args:
            d_model: Embedding dimension
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_seq_length, d_model]
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor with positional encoding [batch_size, seq_len, d_model]
        """
        # Add positional encoding to input
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        """
        Combined token and positional embedding layer.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of token indices [batch_size, seq_len]
            
        Returns:
            Embedded tensor with positional encoding [batch_size, seq_len, d_model]
        """
        # Apply token embedding followed by positional encoding
        return self.positional_encoding(self.token_embedding(x))
