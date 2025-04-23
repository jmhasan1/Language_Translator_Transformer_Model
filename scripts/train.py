import os
import sys
import argparse
import torch
import logging
from torch.utils.data import DataLoader
import time
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import TransformerModel
from data.preprocessing.tokenizer import Tokenizer
from data.preprocessing.dataset_utils import TranslationDataset, collate_batch
from training.trainer import Trainer
from configs.model_config import ModelConfig
from configs.training_config import TrainingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train the transformer translation model")
    
    # Data parameters
    parser.add_argument("--src-data", required=True, help="Path to source language data file")
    parser.add_argument("--tgt-data", required=True, help="Path to target language data file")
    parser.add_argument("--val-src-data", help="Path to validation source language data file")
    parser.add_argument("--val-tgt-data", help="Path to validation target language data file")
    parser.add_argument("--src-tokenizer", help="Path to source tokenizer (for loading pre-trained tokenizer)")
    parser.add_argument("--tgt-tokenizer", help="Path to target tokenizer (for loading pre-trained tokenizer)")
    
    # Model parameters
    parser.add_argument("--d-model", type=int, default=ModelConfig.d_model, help="Model dimension")
    parser.add_argument("--num-heads", type=int, default=ModelConfig.num_heads, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=ModelConfig.num_layers, help="Number of encoder/decoder layers")
    parser.add_argument("--d-ff", type=int, default=ModelConfig.d_ff, help="Feed-forward layer dimension")
    parser.add_argument("--dropout", type=float, default=ModelConfig.dropout, help="Dropout rate")
    parser.add_argument("--max-seq-length", type=int, default=ModelConfig.max_seq_length, help="Maximum sequence length")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size, help="Batch size")
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=TrainingConfig.learning_rate, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=TrainingConfig.weight_decay, help="Weight decay")
    parser.add_argument("--clip-grad-norm", type=float, default=TrainingConfig.clip_grad_norm, help="Gradient clipping norm")
    parser.add_argument("--warmup-steps", type=int, default=TrainingConfig.warmup_steps, help="Warmup steps for learning rate scheduler")
    
    # Output parameters
    parser.add_argument("--save-dir", default="saved_models", help="Directory to save models and tokenizers")
    parser.add_argument("--save-interval", type=int, default=5, help="Save model every N epochs")
    parser.add_argument("--log-interval", type=int, default=100, help="Log training stats every N batches")
    
    # Hardware parameters
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Device to train on (defaults to cuda if available)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision training")
    
    # Resume training
    parser.add_argument("--resume", help="Path to checkpoint to resume training from")
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize tokenizers
    if args.src_tokenizer and args.tgt_tokenizer:
        # Load pre-trained tokenizers
        logger.info("Loading pre-trained tokenizers")
        tokenizer = Tokenizer(
            tokenizer_src_path=args.src_tokenizer,
            tokenizer_tgt_path=args.tgt_tokenizer
        )
    else:
        # Train new tokenizers
        logger.info("Training new tokenizers on the provided data")
        tokenizer = Tokenizer()
        tokenizer.train(
            src_file=args.src_data,
            tgt_file=args.tgt_data,
            vocab_size=ModelConfig.vocab_size
        )
        
        # Save the tokenizers
        src_tokenizer_path = os.path.join(args.save_dir, "tokenizer_src.json")
        tgt_tokenizer_path = os.path.join(args.save_dir, "tokenizer_tgt.json")
        tokenizer.save(src_tokenizer_path, tgt_tokenizer_path)
        logger.info(f"Saved tokenizers to {src_tokenizer_path} and {tgt_tokenizer_path}")
    
    # Get vocabulary sizes
    src_vocab_size = tokenizer.get_src_vocab_size()
    tgt_vocab_size = tokenizer.get_tgt_vocab_size()
    logger.info(f"Source vocabulary size: {src_vocab_size}")
    logger.info(f"Target vocabulary size: {tgt_vocab_size}")
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = TranslationDataset(
        src_file=args.src_data,
        tgt_file=args.tgt_data,
        tokenizer=tokenizer,
        max_len=args.max_seq_length
    )
    
    # Create validation dataset if provided
    val_dataset = None
    if args.val_src_data and args.val_tgt_data:
        val_dataset = TranslationDataset(
            src_file=args.val_src_data,
            tgt_file=args.val_tgt_data,
            tokenizer=tokenizer,
            max_len=args.max_seq_length
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=True if device.type == "cuda" else False
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
            pin_memory=True if device.type == "cuda" else False
        )
    
    # Create model
    logger.info("Initializing transformer model...")
    model = TransformerModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_length=args.max_seq_length,
        dropout=args.dropout
    ).to(device)
    
    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Resumed from epoch {start_epoch}")
        else:
            logger.warning(f"No checkpoint found at {args.resume}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        clip_grad_norm=args.clip_grad_norm,
        warmup_steps=args.warmup_steps,
        mixed_precision=args.mixed_precision
    )
    
    # Save configuration
    config_path = os.path.join(args.save_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Train the model
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        # Train for one epoch
        train_loss = trainer.train_epoch(
            data_loader=train_loader,
            epoch=epoch,
            log_interval=args.log_interval
        )
        
        # Validate
        val_loss = None
        if val_loader:
            val_loss = trainer.evaluate(val_loader)
            logger.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(args.save_dir, "best_model.pt")
                trainer.save_checkpoint(
                    path=best_model_path,
                    epoch=epoch,
                    model_config={
                        'src_vocab_size': src_vocab_size,
                        'tgt_vocab_size': tgt_vocab_size,
                        'd_model': args.d_model,
                        'num_heads': args.num_heads,
                        'num_layers': args.num_layers,
                        'd_ff': args.d_ff,
                        'max_seq_length': args.max_seq_length,
                        'dropout': args.dropout
                    }
                )
                logger.info(f"New best model saved to {best_model_path}")
        else:
            logger.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f}")
        
        # Save checkpoint at specified intervals
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(args.save_dir, f"model_epoch_{epoch+1}.pt")
            trainer.save_checkpoint(
                path=checkpoint_path,
                epoch=epoch,
                model_config={
                    'src_vocab_size': src_vocab_size,
                    'tgt_vocab_size': tgt_vocab_size,
                    'd_model': args.d_model,
                    'num_heads': args.num_heads,
                    'num_layers': args.num_layers,
                    'd_ff': args.d_ff,
                    'max_seq_length': args.max_seq_length,
                    'dropout': args.dropout
                }
            )
            logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        epoch_time = time.time() - start_time
        logger.info(f"Epoch time: {epoch_time:.2f} seconds")
    
    logger.info("Training completed!")
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, "final_model.pt")
    trainer.save_checkpoint(
        path=final_model_path,
        epoch=args.epochs-1,
        model_config={
            'src_vocab_size': src_vocab_size,
            'tgt_vocab_size': tgt_vocab_size,
            'd_model': args.d_model,
            'num_heads': args.num_heads,
            'num_layers': args.num_layers,
            'd_ff': args.d_ff,
            'max_seq_length': args.max_seq_length,
            'dropout': args.dropout
        }
    )
    logger.info(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()
