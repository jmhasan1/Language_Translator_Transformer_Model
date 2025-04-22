import torch
import torch.nn as nn
import math
import os
import json
from typing import Dict, Any, Optional, Union, Tuple

from model.transformer import Transformer
from configs.model_config import ModelConfig


def initialize_weights(model: nn.Module) -> None:
    """
    Initialize weights for the model using Xavier uniform for linear layers
    and normal distribution for embeddings.
    
    Args:
        model: PyTorch model
    """
    if isinstance(model, nn.Linear):
        # Xavier uniform initialization for linear layers
        nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            nn.init.constant_(model.bias, 0)
    elif isinstance(model, nn.Embedding):
        # Initialize embeddings with normal distribution
        nn.init.normal_(model.weight, mean=0, std=model.weight.shape[1] ** -0.5)
        # Special handling for padding token
        if model.padding_idx is not None:
            with torch.no_grad():
                model.weight[model.padding_idx].fill_(0)


def build_model(config: ModelConfig) -> Transformer:
    """
    Build a Transformer model from configuration.
    
    Args:
        config: Model configuration object
        
    Returns:
        Initialized Transformer model
    """
    model = Transformer(
        src_vocab_size=config.src_vocab_size,
        tgt_vocab_size=config.tgt_vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        d_ff=config.d_ff,
        max_seq_length=config.max_seq_length,
        dropout=config.dropout,
        pad_idx=config.pad_idx
    )
    
    # Apply weight initialization
    model.apply(initialize_weights)
    
    return model


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    loss: float,
    save_dir: str,
    config: Dict[str, Any],
    tokenizer_src_path: Optional[str] = None,
    tokenizer_tgt_path: Optional[str] = None,
    name: str = "model"
) -> str:
    """
    Save model checkpoint, optimizer state, and configuration.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch
        loss: Validation loss
        save_dir: Directory to save checkpoint
        config: Model configuration dictionary
        tokenizer_src_path: Path to source tokenizer (if available)
        tokenizer_tgt_path: Path to target tokenizer (if available)
        name: Name prefix for saved files
        
    Returns:
        Path to saved checkpoint
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Build checkpoint dictionary
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'config': config
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if tokenizer_src_path is not None:
        checkpoint['tokenizer_src_path'] = tokenizer_src_path
    
    if tokenizer_tgt_path is not None:
        checkpoint['tokenizer_tgt_path'] = tokenizer_tgt_path
    
    # Save checkpoint
    checkpoint_path = os.path.join(save_dir, f"{name}_checkpoint_epoch{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # Save configuration separately for easy loading
    config_path = os.path.join(save_dir, f"{name}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return checkpoint_path


def load_model(
    checkpoint_path: str,
    device: torch.device = torch.device('cpu')
) -> Tuple[Transformer, Dict[str, Any], Optional[torch.optim.Optimizer], int, float]:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load the model on
        
    Returns:
        Tuple of (model, config, optimizer, epoch, loss)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract configuration
    config = checkpoint['config']
    
    # Build model from config
    model_config = ModelConfig(**config)
    model = build_model(model_config)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Extract other information
    optimizer = None
    if 'optimizer_state_dict' in checkpoint:
        # Note: The optimizer will need to be properly initialized with the model parameters
        # before loading this state dict, which is typically done outside this function
        optimizer_state = checkpoint['optimizer_state_dict']
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    return model, config, optimizer_state if 'optimizer_state_dict' in checkpoint else None, epoch, loss


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    start_token: int,
    end_token: int,
    max_len: int,
    device: torch.device
) -> torch.Tensor:
    """
    Perform greedy decoding for inference.
    
    Args:
        model: Transformer model
        src: Source tensor [batch_size, seq_len]
        start_token: Start token ID
        end_token: End token ID
        max_len: Maximum output sequence length
        device: Device to run inference on
        
    Returns:
        Decoded output sequence [batch_size, seq_len]
    """
    batch_size = src.shape[0]
    
    # Encode the source sequence
    src_mask = model.create_pad_mask(src)
    enc_output = model.encode(src, src_mask)
    
    # Initialize decoder input with start token
    dec_input = torch.ones(batch_size, 1, dtype=torch.long, device=device) * start_token
    
    # Store decoded outputs
    outputs = []
    
    # Decode one token at a time
    for i in range(max_len):
        # Prepare masks
        tgt_mask = None  # Will be created in the model
        memory_mask = src_mask
        
        # Get predictions
        logits = model.decode(dec_input, enc_output, tgt_mask, memory_mask)
        
        # Get next token probabilities (last position only)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # Add predicted token to outputs
        outputs.append(next_token)
        
        # Break if all sequences have end token
        if (next_token == end_token).all():
            break
        
        # Update decoder input for next iteration
        dec_input = torch.cat([dec_input, next_token], dim=1)
    
    # Concatenate all output tokens
    outputs = torch.cat(outputs, dim=1)
    
    return outputs


def create_masks(src, tgt, pad_idx):
    """
    Create necessary masks for transformer input.
    
    Args:
        src: Source tensor [batch_size, src_len]
        tgt: Target tensor [batch_size, tgt_len]
        pad_idx: Padding token index
        
    Returns:
        Tuple of (src_mask, tgt_mask, memory_mask)
    """
    # Create padding mask for source
    src_pad_mask = (src == pad_idx).unsqueeze(1).unsqueeze(2)
    
    # Create padding mask for target
    tgt_pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    
    # Create look-ahead mask for target
    tgt_len = tgt.size(1)
    look_ahead_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
    look_ahead_mask = look_ahead_mask.to(tgt.device)
    
    # Combine padding and look-ahead masks for target
    tgt_mask = tgt_pad_mask | look_ahead_mask.unsqueeze(0)
    
    # Memory mask for encoder-decoder attention (only pad mask from source)
    memory_mask = src_pad_mask
    
    return src_pad_mask, tgt_mask, memory_mask


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss implementation.
    Helps prevent the model from becoming overconfident in its predictions.
    """
    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        """
        Initialize label smoothing loss.
        
        Args:
            smoothing: Smoothing factor (0.0 = no smoothing)
            ignore_index: Index to ignore in loss calculation
        """
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate label smoothing loss.
        
        Args:
            pred: Model predictions [batch_size, seq_len, vocab_size]
            target: Target indices [batch_size, seq_len]
            
        Returns:
            Smoothed loss value
        """
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            # Create a tensor of size [batch_size, seq_len, vocab_size] with smoothing value
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            
            # Fill in the correct class with the confidence value
            true_dist.scatter_(2, target.unsqueeze(2), self.confidence)
            
            # Create mask for ignored indices
            mask = target == self.ignore_index
            mask = mask.unsqueeze(-1).expand_as(true_dist)
            true_dist.masked_fill_(mask, 0.0)
        
        # Calculate loss (sum across sequence dimension, mean across batch dimension)
        loss = torch.sum(-true_dist * pred, dim=-1)
        
        # Mask out ignored positions
        mask = target != self.ignore_index
        loss = (loss * mask).sum() / mask.sum()
        
        return loss


class NoamScheduler:
    """
    Learning rate scheduler for Transformer models.
    Implements the scheduler from "Attention is All You Need".
    """
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer, 
        d_model: int, 
        warmup_steps: int = 4000,
        factor: float = 1.0
    ):
        """
        Initialize Noam scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            d_model: Model dimension
            warmup_steps: Number of warmup steps
            factor: Scaling factor
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0
        self._rate = 0
        
    def step(self):
        """Update learning rate and optimizer step."""
        self._step += 1
        rate = self.get_rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = rate
        self._rate = rate
        
    def get_rate(self) -> float:
        """
        Calculate learning rate according to Noam schedule.
        
        Returns:
            Learning rate value
        """
        arg1 = self._step ** (-0.5)
        arg2 = self._step * (self.warmup_steps ** (-1.5))
        
        return self.factor * (self.d_model ** (-0.5)) * min(arg1, arg2)
