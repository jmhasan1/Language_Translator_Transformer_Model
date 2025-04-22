"""
Configuration file for the training process.
"""

import os
from pathlib import Path

class TrainingConfig:
    def __init__(self,
                 batch_size=32,
                 accumulation_steps=2,
                 epochs=10,
                 lr=0.0001,
                 betas=(0.9, 0.98),
                 eps=1e-9,
                 weight_decay=0.0001,
                 warmup_steps=4000,
                 label_smoothing=0.1,
                 clip_grad_norm=1.0,
                 save_steps=1000,
                 log_steps=100,
                 eval_steps=500,
                 checkpoint_dir="checkpoints",
                 resume_from_checkpoint=None,
                 mixed_precision=True):
        """
        Initialize training configuration with default parameters.
        
        Args:
            batch_size: Training batch size
            accumulation_steps: Gradient accumulation steps
            epochs: Number of training epochs
            lr: Learning rate
            betas: Adam optimizer betas
            eps: Adam optimizer epsilon
            weight_decay: Weight decay for regularization
            warmup_steps: Warmup steps for learning rate scheduler
            label_smoothing: Label smoothing value
            clip_grad_norm: Gradient clipping norm
            save_steps: Steps between checkpoint saves
            log_steps: Steps between logging training metrics
            eval_steps: Steps between evaluations
            checkpoint_dir: Directory to save checkpoints
            resume_from_checkpoint: Path to checkpoint to resume from
            mixed_precision: Whether to use mixed precision training
        """
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.epochs = epochs
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.label_smoothing = label_smoothing
        self.clip_grad_norm = clip_grad_norm
        self.save_steps = save_steps
        self.log_steps = log_steps
        self.eval_steps = eval_steps
        self.checkpoint_dir = checkpoint_dir
        self.resume_from_checkpoint = resume_from_checkpoint
        self.mixed_precision = mixed_precision
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    @classmethod
    def get_low_resource_config(cls):
        """Returns a configuration suitable for lower-end hardware."""
        return cls(
            batch_size=8,
            accumulation_steps=4,
            epochs=5,
            warmup_steps=2000,
            mixed_precision=True
        )

class DataConfig:
    def __init__(self,
                 src_lang="en",
                 tgt_lang="fr",
                 train_data_path="data/datasets/processed/train",
                 val_data_path="data/datasets/processed/val",
                 test_data_path="data/datasets/processed/test",
                 tokenizer_path="data/preprocessing/tokenizer",
                 num_workers=4):
        """
        Initialize data configuration with default parameters.
        
        Args:
            src_lang: Source language code
            tgt_lang: Target language code
            train_data_path: Path to training data
            val_data_path: Path to validation data
            test_data_path: Path to test data
            tokenizer_path: Path to save/load tokenizer
            num_workers: Number of data loader workers
        """
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.tokenizer_path = tokenizer_path
        self.num_workers = num_workers
        
        # Create necessary directories
        for path in [train_data_path, val_data_path, test_data_path, tokenizer_path]:
            os.makedirs(path, exist_ok=True)
