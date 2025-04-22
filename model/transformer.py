import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (persistent state)
        self.register_buffer("pe", pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        
        # Linear projections and reshape
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out_proj(output), attn_weights


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Activation {activation} not supported")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        dropout: float = 0.1,
        pre_norm: bool = True,
        activation: str = "gelu"
    ):
        super().__init__()
        self.pre_norm = pre_norm
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pre_norm:
            # Pre-LN architecture (better training stability)
            attn_input = self.norm1(x)
            attn_output, attn_weights = self.self_attn(attn_input, attn_input, attn_input, mask)
            x = x + self.dropout1(attn_output)
            
            ff_input = self.norm2(x)
            ff_output = self.feed_forward(ff_input)
            x = x + self.dropout2(ff_output)
        else:
            # Post-LN architecture (original transformer)
            attn_output, attn_weights = self.self_attn(x, x, x, mask)
            x = self.norm1(x + self.dropout1(attn_output))
            
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout2(ff_output))
            
        return x, attn_weights


class DecoderLayer(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        dropout: float = 0.1,
        pre_norm: bool = True,
        activation: str = "gelu"
    ):
        super().__init__()
        self.pre_norm = pre_norm
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.pre_norm:
            # Self attention
            attn_input = self.norm1(x)
            self_attn_output, self_attn_weights = self.self_attn(attn_input, attn_input, attn_input, tgt_mask)
            x = x + self.dropout1(self_attn_output)
            
            # Cross attention
            attn_input = self.norm2(x)
            cross_attn_output, cross_attn_weights = self.cross_attn(attn_input, memory, memory, memory_mask)
            x = x + self.dropout2(cross_attn_output)
            
            # Feed forward
            ff_input = self.norm3(x)
            ff_output = self.feed_forward(ff_input)
            x = x + self.dropout3(ff_output)
        else:
            # Self attention
            self_attn_output, self_attn_weights = self.self_attn(x, x, x, tgt_mask)
            x = self.norm1(x + self.dropout1(self_attn_output))
            
            # Cross attention
            cross_attn_output, cross_attn_weights = self.cross_attn(x, memory, memory, memory_mask)
            x = self.norm2(x + self.dropout2(cross_attn_output))
            
            # Feed forward
            ff_output = self.feed_forward(x)
            x = self.norm3(x + self.dropout3(ff_output))
            
        return x, self_attn_weights, cross_attn_weights


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_seq_length: int = 5000,
        dropout: float = 0.1,
        pre_norm: bool = True,
        activation: str = "gelu"
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, pre_norm, activation)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Scale embeddings by sqrt(d_model)
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        attentions = []
        for layer in self.layers:
            x, attn = layer(x, src_mask)
            attentions.append(attn)
            
        if self.layers[0].pre_norm:
            x = self.norm(x)
            
        return x, torch.stack(attentions)


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_seq_length: int = 5000,
        dropout: float = 0.1,
        pre_norm: bool = True,
        activation: str = "gelu"
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, pre_norm, activation)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Scale embeddings by sqrt(d_model)
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        self_attentions = []
        cross_attentions = []
        
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, memory, tgt_mask, memory_mask)
            self_attentions.append(self_attn)
            cross_attentions.append(cross_attn)
            
        if self.layers[0].pre_norm:
            x = self.norm(x)
            
        return x, torch.stack(self_attentions), torch.stack(cross_attentions)


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_seq_length: int = 5000,
        dropout: float = 0.1,
        pre_norm: bool = True,
        activation: str = "gelu",
        tie_embeddings: bool = True,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        
        self.encoder = Encoder(
            src_vocab_size, d_model, num_encoder_layers, num_heads, 
            d_ff, max_seq_length, dropout, pre_norm, activation
        )
        
        self.decoder = Decoder(
            tgt_vocab_size, d_model, num_decoder_layers, num_heads,
            d_ff, max_seq_length, dropout, pre_norm, activation
        )
        
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Optional: Tie decoder embedding and output projection weights
        if tie_embeddings:
            self.output_projection.weight = self.decoder.embedding.weight
            
        self.label_smoothing = label_smoothing
        
    def encode(
        self, 
        src: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(src, src_mask)
    
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.decoder(tgt, memory, tgt_mask, memory_mask)
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        memory, _ = self.encode(src, src_mask)
        decoder_output, _, _ = self.decode(tgt, memory, tgt_mask, memory_mask)
        output = self.output_projection(decoder_output)
        return output
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate a square mask for the sequence.
        
        The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def create_pad_mask(self, seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
        """Create mask for padding tokens."""
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.criterion = nn.KLDivLoss(reduction='sum')
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: [batch_size, seq_len, vocab_size]
        # target: [batch_size, seq_len]
        
        batch_size, seq_len, vocab_size = pred.size()
        pred = pred.contiguous().view(-1, vocab_size)
        target = target.contiguous().view(-1)
        
        # Create mask for ignored indices
        non_pad_mask = target != self.ignore_index
        n_tokens = non_pad_mask.sum()
        
        # Create smoothed targets
        target_flat = target.view(-1)
        valid_indices = non_pad_mask.view(-1)
        
        # Only consider non-ignored indices
        pred_valid = pred[valid_indices]
        target_valid = target_flat[valid_indices]
        
        # Create label smoothed target distributions
        smoothed_targets = torch.zeros_like(pred_valid)
        smoothed_targets.fill_(self.smoothing / (vocab_size - 1))
        smoothed_targets.scatter_(1, target_valid.unsqueeze(1), 1.0 - self.smoothing)
        
        # Apply KL divergence loss
        log_pred = F.log_softmax(pred_valid, dim=-1)
        loss = self.criterion(log_pred, smoothed_targets) / n_tokens
        
        return loss
