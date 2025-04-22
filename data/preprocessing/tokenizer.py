import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import sentencepiece as spm
import pandas as pd
from sklearn.model_selection import train_test_split


class TokenizerSentencePiece:
    def __init__(
        self,
        vocab_size: int = 8000,
        model_type: str = "bpe",
        character_coverage: float = 0.9995,
        model_prefix: str = "tokenizer"
    ):
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.character_coverage = character_coverage
        self.model_prefix = model_prefix
        
        # Special tokens
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        
        # IDs will be set after training
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        # SentencePiece processor
        self.sp = None
        
    def train(self, texts: List[str], output_dir: str = "./"):
        """Train a SentencePiece tokenizer on the provided texts."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Write texts to a temporary file
        corpus_file = os.path.join(output_dir, "corpus.txt")
        with open(corpus_file, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text + "\n")
                
        # Train SentencePiece model
        model_path = os.path.join(output_dir, self.model_prefix)
        spm.SentencePieceTrainer.train(
            input=corpus_file,
            model_prefix=model_path,
            vocab_size=self.vocab_size,
            model_type=self.model_type,
            character_coverage=self.character_coverage,
            pad_id=self.pad_token_id,
            unk_id=self.unk_token_id,
            bos_id=self.bos_token_id,
            eos_id=self.eos_token_id,
            pad_piece=self.pad_token,
            unk_piece=self.unk_token,
            bos_piece=self.bos_token,
            eos_piece=self.eos_token
        )
        
        # Load the trained model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f"{model_path}.model")
        
        # Clean up temporary file
        os.remove(corpus_file)
        
        print(f"Tokenizer trained and saved to {model_path}.model and {model_path}.vocab")
        
    def load(self, model_path: str):
        """Load a trained SentencePiece model."""
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        if self.sp is None:
            raise ValueError("Tokenizer model not loaded. Call train() or load() first.")
            
        if add_special_tokens:
            tokens = [self.bos_token_id] + self.sp.encode(text) + [self.eos_token_id]
        else:
            tokens = self.sp.encode(text)
            
        return tokens
    
    def decode(self, token_ids: List[int], remove_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if self.sp is None:
            raise ValueError("Tokenizer model not loaded. Call train() or load() first.")
            
        if remove_special_tokens:
            # Filter out special tokens
            token_ids = [
                token_id for token_id in token_ids 
                if token_id not in [self.pad_token_id, self.bos_token_id, self.eos_token_id]
            ]
            
        return self.sp.decode(token_ids)
    
    def save(self, path: str):
        """Save tokenizer files to the specified path."""
        if self.sp is None:
            raise ValueError("Tokenizer model not loaded. Call train() or load() first.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Copy the model and vocab files
        model_path = f"{self.model_prefix}.model"
        vocab_path = f"{self.model_prefix}.vocab"
        
        import shutil
        shutil.copy(model_path, f"{path}.model")
        shutil.copy(vocab_path, f"{path}.vocab")
        
    @property
    def vocab_size(self):
        """Get the vocabulary size."""
        if self.sp is not None:
            return self.sp.get_piece_size()
        return self._vocab_size
    
    @vocab_size.setter
    def vocab_size(self, value):
        """Set the vocabulary size."""
        self._vocab_size = value


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_texts: List[str],
        tgt_texts: List[str],
        src_tokenizer: TokenizerSentencePiece,
        tgt_tokenizer: TokenizerSentencePiece,
        max_length: int = 128
    ):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        
        # Tokenize
        src_tokens = self.src_tokenizer.encode(src_text)
        tgt_tokens = self.tgt_tokenizer.encode(tgt_text)
        
        # Truncate if needed
        src_tokens = src_tokens[:self.max_length]
        tgt_tokens = tgt_tokens[:self.max_length]
        
        # Convert to tensors
        src_tensor = torch.tensor(src_tokens, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_tokens, dtype=torch.long)
        
        return {"src": src_tensor, "tgt": tgt_tensor}


def prepare_datasets(
    data_path: str,
    src_lang: str,
    tgt_lang: str,
    src_tokenizer: TokenizerSentencePiece,
    tgt_tokenizer: TokenizerSentencePiece,
    max_length: int = 128,
    batch_size: int = 32,
    test_size: float = 0.1,
    val_size: float = 0.1,
    train_tokenizer: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare datasets and dataloaders for training."""
    # Load data
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.tsv'):
        df = pd.read_csv(data_path, sep='\t')
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
        
    src_texts = df[src_lang].tolist()
    tgt_texts = df[tgt_lang].tolist()
    
    # Train tokenizers if requested
    if train_tokenizer:
        print(f"Training source language tokenizer ({src_lang})...")
        src_tokenizer.train(src_texts, output_dir=f"./tokenizers/{src_lang}")
        
        print(f"Training target language tokenizer ({tgt_lang})...")
        tgt_tokenizer.train(tgt_texts, output_dir=f"./tokenizers/{tgt_lang}")
    
    # Split data into train, validation, and test sets
    train_src, temp_src, train_tgt, temp_tgt = train_test_split(
        src_texts, tgt_texts, test_size=test_size+val_size, random_state=42
    )
    
    val_ratio = val_size / (test_size + val_size)
    val_src, test_src, val_tgt, test_tgt = train_test_split(
        temp_src, temp_tgt, test_size=1-val_ratio, random_state=42
    )
    
    # Create datasets
    train_dataset = TranslationDataset(
        train_src, train_tgt, src_tokenizer, tgt_tokenizer, max_length
    )
    
    val_dataset = TranslationDataset(
        val_src, val_tgt, src_tokenizer, tgt_tokenizer, max_length
    )
    
    test_dataset = TranslationDataset(
        test_src, test_tgt, src_tokenizer, tgt_tokenizer, max_length
    )
    
    # Create dataloaders
    def collate_fn(batch):
        src_tensors = [item["src"] for item in batch]
        tgt_tensors = [item["tgt"] for item in batch]
        
        # Pad sequences
        src_tensors = torch.nn.utils.rnn.pad_sequence(
            src_tensors, batch_first=True, padding_value=src_tokenizer.pad_token_id
        )
        
        tgt_tensors = torch.nn.utils.rnn.pad_sequence(
            tgt_tensors, batch_first=True, padding_value=tgt_tokenizer.pad_token_id
        )
        
        return {"src": src_tensors, "tgt": tgt_tensors}
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_fn
    )
    
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate_fn
    )
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    return train_dataloader, val_dataloader, test_dataloader
