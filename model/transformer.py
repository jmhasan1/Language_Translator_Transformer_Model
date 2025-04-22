import torch
import torch.nn as nn
from typing import Optional

from model.layers.encoder import Encoder
from model.layers.decoder import Decoder


class Transformer(nn.Module):
    """
    A complete Transformer model for sequence-to-sequence translation tasks.
    Combines encoder and decoder components for end-to-end translation.
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 100,
        dropout: float = 0.1,
        pad_idx: int = 0
    ):
        """
        Initialize the Transformer model.
        
        Args:
            src_vocab_size: Size of the source language vocabulary
            tgt_vocab_size: Size of the target language vocabulary
            d_model: Dimension of the model (embedding dimension)
            n_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            d_ff: Dimension of the feed-forward network
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
            pad_idx: Padding token index
        """
        super(Transformer, self).__init__()
        
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Create encoder and decoder components
        self.encoder = Encoder(
            src_vocab_size=src_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_encoder_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout,
            pad_idx=pad_idx
        )
        
        self.decoder = Decoder(
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_decoder_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout,
            pad_idx=pad_idx
        )
        
        # Final linear layer to project to vocabulary size
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
        
    def create_pad_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Create padding mask for sequences.
        
        Args:
            seq: Input sequence tensor [batch_size, seq_len]
            
        Returns:
            Padding mask [batch_size, 1, 1, seq_len]
        """
        # Create a mask for padding tokens (1 for pad tokens, 0 for non-pad)
        pad_mask = (seq == self.pad_idx).unsqueeze(1).unsqueeze(2)
        return pad_mask
    
    def create_look_ahead_mask(self, seq_len: int) -> torch.Tensor:
        """
        Create look-ahead mask for decoder self-attention.
        
        Args:
            seq_len: Length of the sequence
            
        Returns:
            Look-ahead mask [seq_len, seq_len]
        """
        # Create a lower triangular matrix with 0s in the lower triangle and 1s elsewhere
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask
    
    def forward(
        self, 
        src: torch.Tensor, 
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the transformer.
        
        Args:
            src: Source sequence [batch_size, src_len]
            tgt: Target sequence [batch_size, tgt_len]
            src_mask: Source padding mask
            tgt_mask: Target padding mask combined with look-ahead mask
            memory_mask: Mask for encoder-decoder attention
            
        Returns:
            Output logits [batch_size, tgt_len, tgt_vocab_size]
        """
        # Create masks if not provided
        if src_mask is None:
            src_mask = self.create_pad_mask(src)
        
        if tgt_mask is None:
            # Create padding mask
            tgt_pad_mask = self.create_pad_mask(tgt)
            
            # Create look-ahead mask
            tgt_len = tgt.size(1)
            look_ahead_mask = self.create_look_ahead_mask(tgt_len).to(tgt.device)
            
            # Combine padding and look-ahead masks
            # A position is masked if either mask is True (logical OR)
            tgt_mask = tgt_pad_mask | look_ahead_mask.unsqueeze(0)
        
        if memory_mask is None:
            # For encoder-decoder attention, we only need to mask padding in the source
            memory_mask = self.create_pad_mask(src)
        
        # Get encoder output
        enc_output = self.encoder(src, src_mask)
        
        # Pass through decoder with encoder output as memory
        dec_output = self.decoder(tgt, enc_output, tgt_mask, memory_mask)
        
        # Project to vocabulary size
        logits = self.output_linear(dec_output)
        
        return logits
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode the source sequence.
        
        Args:
            src: Source sequence [batch_size, src_len]
            src_mask: Source padding mask
            
        Returns:
            Encoder output [batch_size, src_len, d_model]
        """
        if src_mask is None:
            src_mask = self.create_pad_mask(src)
            
        return self.encoder(src, src_mask)
    
    def decode(
        self, 
        tgt: torch.Tensor, 
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode with target sequence and encoder memory.
        
        Args:
            tgt: Target sequence [batch_size, tgt_len]
            memory: Encoder output [batch_size, src_len, d_model]
            tgt_mask: Target mask
            memory_mask: Memory mask for encoder-decoder attention
            
        Returns:
            Decoder output logits [batch_size, tgt_len, tgt_vocab_size]
        """
        # Create masks if not provided
        if tgt_mask is None:
            # Create padding mask
            tgt_pad_mask = self.create_pad_mask(tgt)
            
            # Create look-ahead mask
            tgt_len = tgt.size(1)
            look_ahead_mask = self.create_look_ahead_mask(tgt_len).to(tgt.device)
            
            # Combine padding and look-ahead masks
            tgt_mask = tgt_pad_mask | look_ahead_mask.unsqueeze(0)
        
        # Decode and project to vocabulary
        dec_output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        logits = self.output_linear(dec_output)
        
        return logits
