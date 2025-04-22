"""
Configuration file for the transformer model architecture.
"""

class ModelConfig:
    def __init__(self, 
                 src_vocab_size=32000,
                 tgt_vocab_size=32000,
                 d_model=512,
                 num_heads=8, 
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 d_ff=2048,
                 dropout=0.1,
                 max_seq_length=128,
                 pad_idx=0,
                 share_embeddings=True,
                 share_decoder_embeddings=True):
        """
        Initialize model configuration with default parameters suitable for a small machine.
        
        Args:
            src_vocab_size: Size of source vocabulary
            tgt_vocab_size: Size of target vocabulary
            d_model: Dimension of model (embedding dimension)
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            d_ff: Dimension of feed-forward network
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
            pad_idx: Padding token index
            share_embeddings: Whether to share embeddings between encoder and decoder
            share_decoder_embeddings: Whether to share embeddings between decoder input and output
        """
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        self.pad_idx = pad_idx
        self.share_embeddings = share_embeddings
        self.share_decoder_embeddings = share_decoder_embeddings

    @classmethod
    def get_small_config(cls):
        """Returns a smaller model configuration suitable for lower-end hardware."""
        return cls(
            src_vocab_size=16000,
            tgt_vocab_size=16000,
            d_model=256,
            num_heads=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
            d_ff=1024,
            max_seq_length=64
        )
    
    @classmethod
    def get_base_config(cls):
        """Returns the base model configuration (similar to transformer base)."""
        return cls()
    
    @classmethod
    def get_large_config(cls):
        """Returns a larger model configuration (only for powerful hardware)."""
        return cls(
            d_model=768,
            num_heads=12,
            num_encoder_layers=8,
            num_decoder_layers=8,
            d_ff=3072
        )
