"""
Tokenizer implementation for the language translation model.
"""

import os
import logging
from typing import List, Dict, Union, Tuple, Optional
import sentencepiece as spm

logger = logging.getLogger(__name__)

class Tokenizer:
    def __init__(self, model_path: str = None):
        """
        Initialize tokenizer from a model or prepare for training.
        
        Args:
            model_path: Path to a trained SentencePiece model
        """
        self.sp_model = None
        self.model_path = model_path
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3
        }
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def train(self, 
              input_file: str, 
              vocab_size: int = 32000,
              model_type: str = "bpe",
              character_coverage: float = 0.9995,
              model_prefix: str = "tokenizer",
              input_sentence_size: int = 1000000,
              shuffle_input_sentence: bool = True,
              normalization_rule_name: str = "nmt_nfkc_cf",
              add_dummy_prefix: bool = True) -> None:
        """
        Train a SentencePiece tokenizer on input text file.
        
        Args:
            input_file: Path to input text file (one sentence per line)
            vocab_size: Size of vocabulary
            model_type: SentencePiece model type (bpe, unigram, char, word)
            character_coverage: Character coverage
            model_prefix: Output model prefix
            input_sentence_size: Number of sentences to sample for training
            shuffle_input_sentence: Whether to shuffle sentences
            normalization_rule_name: Normalization rule
            add_dummy_prefix: Whether to add dummy prefix
        """
        logger.info(f"Training SentencePiece tokenizer on {input_file} with vocab size {vocab_size}")
        
        # Ensure model directory exists
        os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
        
        # Train model
        spm.SentencePieceTrainer.train(
            input=input_file,
            vocab_size=vocab_size - len(self.special_tokens),  # Account for special tokens
            model_prefix=model_prefix,
            model_type=model_type,
            character_coverage=character_coverage,
            input_sentence_size=input_sentence_size,
            shuffle_input_sentence=shuffle_input_sentence,
            normalization_rule_name=normalization_rule_name,
            pad_id=self.special_tokens["<pad>"],
            unk_id=self.special_tokens["<unk>"],
            bos_id=self.special_tokens["<bos>"],
            eos_id=self.special_tokens["<eos>"],
            user_defined_symbols=list(self.special_tokens.keys()),
            add_dummy_prefix=add_dummy_prefix
        )
        
        # Load the trained model
        self.model_path = f"{model_prefix}.model"
        self.load(self.model_path)
        logger.info(f"Tokenizer trained and saved to {self.model_path}")
    
    def load(self, model_path: str) -> None:
        """
        Load a trained SentencePiece model.
        
        Args:
            model_path: Path to SentencePiece model
        """
        logger.info(f"Loading tokenizer from {model_path}")
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)
        self.model_path = model_path
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        if self.sp_model is None:
            logger.error("Tokenizer model not loaded")
            raise ValueError("Tokenizer model not loaded. Call load() or train() first.")
        
        tokens = self.sp_model.encode(text, out_type=int)
        
        if add_special_tokens:
            tokens = [self.special_tokens["<bos>"]] + tokens + [self.special_tokens["<eos>"]]
        
        return tokens
    
    def decode(self, token_ids: List[int], remove_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            remove_special_tokens: Whether to remove special tokens
            
        Returns:
            Decoded text
        """
        if self.sp_model is None:
            logger.error("Tokenizer model not loaded")
            raise ValueError("Tokenizer model not loaded. Call load() or train() first.")
        
        if remove_special_tokens:
            special_ids = list(self.special_tokens.values())
            token_ids = [t for t in token_ids if t not in special_ids]
        
        return self.sp_model.decode(token_ids)
    
    def get_vocab_size(self) -> int:
        """
        Get the vocabulary size.
        
        Returns:
            Vocabulary size including special tokens
        """
        if self.sp_model is None:
            logger.warning("Tokenizer model not loaded, returning only special tokens count")
            return len(self.special_tokens)
        
        return self.sp_model.get_piece_size()
    
    def token_to_id(self, token: str) -> int:
        """
        Convert a token to its ID.
        
        Args:
            token: The token string
            
        Returns:
            Token ID
        """
        if token in self.special_tokens:
            return self.special_tokens[token]
        
        if self.sp_model is None:
            logger.error("Tokenizer model not loaded")
            raise ValueError("Tokenizer model not loaded. Call load() or train() first.")
        
        return self.sp_model.piece_to_id(token)
    
    def id_to_token(self, token_id: int) -> str:
        """
        Convert an ID to its token.
        
        Args:
            token_id: The token ID
            
        Returns:
            Token string
        """
        for token, idx in self.special_tokens.items():
            if idx == token_id:
                return token
        
        if self.sp_model is None:
            logger.error("Tokenizer model not loaded")
            raise ValueError("Tokenizer model not loaded. Call load() or train() first.")
        
        return self.sp_model.id_to_piece(token_id)
    
    def batch_encode(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """
        Encode a batch of texts to token IDs.
        
        Args:
            texts: List of input texts
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of lists of token IDs
        """
        return [self.encode(text, add_special_tokens) for text in texts]
    
    def batch_decode(self, batch_token_ids: List[List[int]], remove_special_tokens: bool = True) -> List[str]:
        """
        Decode a batch of token IDs to texts.
        
        Args:
            batch_token_ids: List of lists of token IDs
            remove_special_tokens: Whether to remove special tokens
            
        Returns:
            List of decoded texts
        """
        return [self.decode(token_ids, remove_special_tokens) for token_ids in batch_token_ids]


class TranslationTokenizer:
    def __init__(self, 
                src_tokenizer_path: Optional[str] = None, 
                tgt_tokenizer_path: Optional[str] = None,
                share_tokenizer: bool = False):
        """
        Initialize tokenizers for source and target languages.
        
        Args:
            src_tokenizer_path: Path to source language tokenizer
            tgt_tokenizer_path: Path to target language tokenizer
            share_tokenizer: Use same tokenizer for source and target
        """
        self.share_tokenizer = share_tokenizer
        
        if share_tokenizer:
            # Use the same tokenizer for both source and target
            self.src_tokenizer = Tokenizer(src_tokenizer_path)
            self.tgt_tokenizer = self.src_tokenizer
        else:
            # Use separate tokenizers
            self.src_tokenizer = Tokenizer(src_tokenizer_path)
            self.tgt_tokenizer = Tokenizer(tgt_tokenizer_path)
    
    def train(self, 
              src_file: str, 
              tgt_file: Optional[str] = None, 
              src_vocab_size: int = 32000,
              tgt_vocab_size: int = 32000,
              src_prefix: str = "tokenizer_src",
              tgt_prefix: str = "tokenizer_tgt",
              **kwargs) -> None:
        """
        Train tokenizers for source and target languages.
        
        Args:
            src_file: Path to source language text file
            tgt_file: Path to target language text file (optional if share_tokenizer)
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            src_prefix: Output model prefix for source tokenizer
            tgt_prefix: Output model prefix for target tokenizer
            **kwargs: Additional arguments for Tokenizer.train()
        """
        logger.info(f"Training source tokenizer on {src_file}")
        self.src_tokenizer.train(src_file, vocab_size=src_vocab_size, model_prefix=src_prefix, **kwargs)
        
        if self.share_tokenizer:
            logger.info("Using shared tokenizer for source and target")
            self.tgt_tokenizer = self.src_tokenizer
        else:
            if tgt_file is None:
                raise ValueError("Target file must be provided when not sharing tokenizers")
            
            logger.info(f"Training target tokenizer on {tgt_file}")
            self.tgt_tokenizer.train(tgt_file, vocab_size=tgt_vocab_size, model_prefix=tgt_prefix, **kwargs)
    
    def get_src_vocab_size(self) -> int:
        """Get source vocabulary size."""
        return self.src_tokenizer.get_vocab_size()
    
    def get_tgt_vocab_size(self) -> int:
        """Get target vocabulary size."""
        return self.tgt_tokenizer.get_vocab_size()
    
    def encode_src(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode source text."""
        return self.src_tokenizer.encode(text, add_special_tokens)
    
    def encode_tgt(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode target text."""
        return self.tgt_tokenizer.encode(text, add_special_tokens)
    
    def decode_src(self, token_ids: List[int], remove_special_tokens: bool = True) -> str:
        """Decode source token IDs."""
        return self.src_tokenizer.decode(token_ids, remove_special_tokens)
    
    def decode_tgt(self, token_ids: List[int], remove_special_tokens: bool = True) -> str:
        """Decode target token IDs."""
        return self.tgt_tokenizer.decode(token_ids, remove_special_tokens)
    
    def get_pad_id(self) -> int:
        """Get padding token ID."""
        return self.src_tokenizer.special_tokens["<pad>"]
    
    def get_bos_id(self) -> int:
        """Get beginning of sequence token ID."""
        return self.src_tokenizer.special_tokens["<bos>"]
    
    def get_eos_id(self) -> int:
        """Get end of sequence token ID."""
        return self.src_tokenizer.special_tokens["<eos>"]
