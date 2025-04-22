"""
Dataset and data loading utilities for the language translation model.
"""

import os
import logging
import random
from typing import List, Dict, Tuple, Iterator, Optional, Union, Callable
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from data.preprocessing.tokenizer import TranslationTokenizer

logger = logging.getLogger(__name__)

class TranslationDataset(Dataset):
    def __init__(self, 
                 source_data: List[List[int]],
                 target_data: List[List[int]],
                 pad_idx: int = 0,
                 max_seq_length: int = 128):
        """
        Dataset for machine translation with tokenized data.
        
        Args:
            source_data: List of lists of source token IDs
            target_data: List of lists of target token IDs
            pad_idx: Padding token index
            max_seq_length: Maximum sequence length
        """
        self.source_data = source_data
        self.target_data = target_data
        self.pad_idx = pad_idx
        self.max_seq_length = max_seq_length
        
        assert len(source_data) == len(target_data), "Source and target data must have the same length"
    
    def __len__(self) -> int:
        return len(self.source_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with 'src', 'tgt_input', 'tgt_output', 'src_mask', 'tgt_mask', 'src_padding_mask', 'tgt_padding_mask'
        """
        src_tokens = self.source_data[idx][:self.max_seq_length]
        tgt_tokens = self.target_data[idx][:self.max_seq_length]
        
        # Create source tensor
        src = torch.tensor(src_tokens, dtype=torch.long)
        
        # Create target input (remove EOS) and output (remove BOS) tensors
        tgt_input = torch.tensor(tgt_tokens[:-1] if len(tgt_tokens) > 1 else tgt_tokens, dtype=torch.long)
        tgt_output = torch.tensor(tgt_tokens[1:] if len(tgt_tokens) > 1 else tgt_tokens, dtype=torch.long)
        
        # Create padding masks
        src_padding_mask = (src != self.pad_idx).unsqueeze(0)  # [1, seq_len]
        tgt_padding_mask = (tgt_input != self.pad_idx).unsqueeze(0)  # [1, seq_len]
        
        return {
            'src': src,
            'tgt_input': tgt_input,
            'tgt_output': tgt_output,
            'src_padding_mask': src_padding_mask,
            'tgt_padding_mask': tgt_padding_mask
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_idx: int = 0) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader to handle variable-length sequences.
    
    Args:
        batch: List of samples
        pad_idx: Padding token index
        
    Returns:
        Dictionary with batched tensors
    """
    max_src_len = max(item['src'].size(0) for item in batch)
    max_tgt_input_len = max(item['tgt_input'].size(0) for item in batch)
    max_tgt_output_len = max(item['tgt_output'].size(0) for item in batch)
    
    # Prepare tensors
    batch_size = len(batch)
    src = torch.full((batch_size, max_src_len), pad_idx, dtype=torch.long)
    tgt_input = torch.full((batch_size, max_tgt_input_len), pad_idx, dtype=torch.long)
    tgt_output = torch.full((batch_size, max_tgt_output_len), pad_idx, dtype=torch.long)
    src_padding_mask = torch.zeros((batch_size, 1, max_src_len), dtype=torch.bool)
    tgt_padding_mask = torch.zeros((batch_size, 1, max_tgt_input_len), dtype=torch.bool)
    
    # Fill tensors
    for i, item in enumerate(batch):
        src_len = item['src'].size(0)
        tgt_input_len = item['tgt_input'].size(0)
        tgt_output_len = item['tgt_output'].size(0)
        
        src[i, :src_len] = item['src']
        tgt_input[i, :tgt_input_len] = item['tgt_input']
        tgt_output[i, :tgt_output_len] = item['tgt_output']
        src_padding_mask[i, :, :src_len] = item['src_padding_mask']
        tgt_padding_mask[i, :, :tgt_input_len] = item['tgt_padding_mask']
    
    # Generate attention masks for transformer
    # Source attention mask (not needed for encoder self-attention due to padding mask)
    src_mask = None
    
    # Target attention mask (needed for decoder to prevent attending to future tokens)
    tgt_mask = torch.triu(
        torch.ones((max_tgt_input_len, max_tgt_input_len), dtype=torch.bool),
        diagonal=1
    )
    
    return {
        'src': src,
        'tgt_input': tgt_input,
        'tgt_output': tgt_output,
        'src_mask': src_mask,
        'tgt_mask': tgt_mask,
        'src_padding_mask': src_padding_mask,
        'tgt_padding_mask': tgt_padding_mask
    }


def prepare_dataset(src_file: str,
                   tgt_file: str,
                   tokenizer: TranslationTokenizer,
                   max_seq_length: int = 128,
                   output_dir: Optional[str] = None,
                   split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                   seed: int = 42) -> Dict[str, TranslationDataset]:
    """
    Prepare datasets from source and target files.
    
    Args:
        src_file: Path to source text file (one sentence per line)
        tgt_file: Path to target text file (one sentence per line)
        tokenizer: Translation tokenizer
        max_seq_length: Maximum sequence length
        output_dir: Directory to save processed data (optional)
        split_ratios: Train, validation, test split ratios
        seed: Random seed for data splitting
        
    Returns:
        Dictionary with 'train', 'val', 'test' datasets
    """
    random.seed(seed)
    
    logger.info(f"Preparing dataset from {src_file} and {tgt_file}")
    
    # Read files
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f if line.strip()]
    
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_lines = [line.strip() for line in f if line.strip()]
    
    assert len(src_lines) == len(tgt_lines), "Source and target files must have the same number of lines"
    
    # Tokenize data
    logger.info("Tokenizing data...")
    src_tokenized = []
    tgt_tokenized = []
    
    for src_line, tgt_line in tqdm(zip(src_lines, tgt_lines), total=len(src_lines), desc="Tokenizing"):
        src_tokens = tokenizer.encode_src(src_line)
        tgt_tokens = tokenizer.encode_tgt(tgt_line)
        
        # Skip examples that would be too long after tokenization
        if len(src_tokens) <= max_seq_length and len(tgt_tokens) <= max_seq_length:
            src_tokenized.append(src_tokens)
            tgt_tokenized.append(tgt_tokens)
    
    # Split data
    logger.info(f"Splitting data with ratios {split_ratios}...")
    data_size = len(src_tokenized)
    indices = list(range(data_size))
    random.shuffle(indices)
    
    train_size = int(split_ratios[0] * data_size)
    val_size = int(split_ratios[1] * data_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create datasets
    train_dataset = TranslationDataset(
        [src_tokenized[i] for i in train_indices],
        [tgt_tokenized[i] for i in train_indices],
        pad_idx=tokenizer.get_pad_id(),
        max_seq_length=max_seq_length
    )
    
    val_dataset = TranslationDataset(
        [src_tokenized[i] for i in val_indices],
        [tgt_tokenized[i] for i in val_indices],
        pad_idx=tokenizer.get_pad_id(),
        max_seq_length=max_seq_length
    )
    
    test_dataset = TranslationDataset(
        [src_tokenized[i] for i in test_indices],
        [tgt_tokenized[i] for i in test_indices],
        pad_idx=tokenizer.get_pad_id(),
        max_seq_length=max_seq_length
    )
    
    logger.info(f"Created datasets: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    # Save processed data if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving processed data to {output_dir}")
        
        # Save dataset statistics
        with open(os.path.join(output_dir, 'stats.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Total examples: {data_size}\n")
            f.write(f"Train examples: {len(train_dataset)}\n")
            f.write(f"Validation examples: {len(val_dataset)}\n")
            f.write(f"Test examples: {len(test_dataset)}\n")
            f.write(f"Source vocabulary size: {tokenizer.get_src_vocab_size()}\n")
            f.write(f"Target vocabulary size: {tokenizer.get_tgt_vocab_size()}\n")
    
    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }


def get_dataloader(dataset: TranslationDataset, 
                   batch_size: int = 32,
                   shuffle: bool = True,
                   num_workers: int = 4,
                   pad_idx: int = 0) -> DataLoader:
    """
    Create a DataLoader for a TranslationDataset.
    
    Args:
        dataset: Translation dataset
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        pad_idx: Padding token index
        
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
        pin_memory=True
    )


def download_wmt_dataset(output_dir: str, language_pair: str = "en-fr", year: str = "14"):
    """
    Download WMT dataset for a language pair.
    
    Args:
        output_dir: Directory to save downloaded data
        language_pair: Language pair code (e.g., 'en-fr', 'en-de')
        year: Dataset year (e.g., '14' for WMT14)
    """
    try:
        from torchtext.data.datasets import WMT14
        import urllib.request
        import tarfile
    except ImportError:
        logger.error("Failed to import required packages for downloading WMT dataset")
        raise
    
    os.makedirs(output_dir, exist_ok=True)
    src_lang, tgt_lang = language_pair.split('-')
    
    logger.info(f"Downloading WMT{year} {language_pair} dataset...")
    
    # This is a simplified version that may not work for all WMT years and language pairs
    # For a real implementation, use torchtext or huggingface datasets
    base_url = f"https://www.statmt.org/wmt{year}/training-parallel-nc-v{year}"
    filename = f"{src_lang}-{tgt_lang}.tgz"
    url = f"{base_url}/{filename}"
    
    try:
        # Download file
        logger.info(f"Downloading from {url}")
        filepath = os.path.join(output_dir, filename)
        urllib.request.urlretrieve(url, filepath)
        
        # Extract file
        logger.info(f"Extracting {filepath}")
        with tarfile.open(filepath) as tar:
            tar.extractall(path=output_dir)
        
        logger.info(f"Dataset downloaded and extracted to {output_dir}")
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        logger.info("Please download the dataset manually or use another dataset source.")


def preprocess_and_tokenize(src_file: str, 
                           tgt_file: str,
                           output_dir: str,
                           src_lang: str = "en",
                           tgt_lang: str = "fr",
                           vocab_size: int = 32000,
                           max_seq_length: int = 128,
                           share_tokenizer: bool = False):
    """
    Preprocess and tokenize data files for translation.
    
    Args:
        src_file: Source language file
        tgt_file: Target language file
        output_dir: Output directory for processed data
        src_lang: Source language code
        tgt_lang: Target language code
        vocab_size: Vocabulary size
        max_seq_length: Maximum sequence length
        share_tokenizer: Whether to share tokenizer between source and target
    """
    logger.info(f"Preprocessing {src_lang}-{tgt_lang} data")
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    # Paths for tokenizer models
    src_tokenizer_path = os.path.join(tokenizer_dir, f"tokenizer_{src_lang}")
    tgt_tokenizer_path = os.path.join(tokenizer_dir, f"tokenizer_{tgt_lang}")
    
    # Initialize and train tokenizer
    tokenizer = TranslationTokenizer(share_tokenizer=share_tokenizer)
    tokenizer.train(
        src_file=src_file,
        tgt_file=None if share_tokenizer else tgt_file,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        src_prefix=src_tokenizer_path,
        tgt_prefix=tgt_tokenizer_path
    )
    
    # Prepare datasets
    datasets = prepare_dataset(
        src_file=src_file,
        tgt_file=tgt_file,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        output_dir=output_dir
    )
    
    logger.info("Preprocessing and tokenization completed")
    return tokenizer, datasets