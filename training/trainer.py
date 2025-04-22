import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from typing import Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class NoamScheduler:
    """Implements the Noam learning rate schedule from the Transformer paper."""
    
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer, 
        d_model: int, 
        warmup_steps: int = 4000,
        factor: float = 1.0
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0
        self._rate = 0
        
    def step(self):
        """Update learning rate and take optimizer step."""
        self._step += 1
        rate = self._get_lr_scale()
        self._rate = rate
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = rate
            
    def _get_lr_scale(self) -> float:
        """Calculate learning rate scale based on current step."""
        step = max(1, self._step)  # Avoid division by zero
        scale = self.factor * (self.d_model ** -0.5) * min(
            step ** -0.5, step * (self.warmup_steps ** -1.5)
        )
        return scale
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self._rate


class TranslationTrainer:
    def __init__(
        self,
        model: nn.Module,
        src_tokenizer,
        tgt_tokenizer,
        device: torch.device,
        config: Dict
    ):
        self.model = model
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.device = device
        self.config = config
        
        # Set model to device
        self.model.to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config['learning_rate'], 
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Initialize learning rate scheduler
        self.scheduler = NoamScheduler(
            self.optimizer, 
            config['d_model'], 
            config['warmup_steps']
        )
        
        # Initialize loss function with label smoothing
        if config['label_smoothing'] > 0:
            self.criterion = LabelSmoothingLoss(
                smoothing=config['label_smoothing'], 
                ignore_index=self.tgt_tokenizer.pad_token_id
            )
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.tgt_tokenizer.pad_token_id)
            
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self, train_dataloader: DataLoader) -> float:
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            
            # Create masks
            src_mask = self.model.create_pad_mask(src, self.src_tokenizer.pad_token_id).to(self.device)
            tgt_mask = self.model.create_pad_mask(tgt[:, :-1], self.tgt_tokenizer.pad_token_id).to(self.device)
            tgt_attn_mask = self.model.generate_square_subsequent_mask(tgt.size(1) - 1).to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(
                src=src,
                tgt=tgt[:, :-1],
                src_mask=src_mask,
                tgt_mask=tgt_mask & tgt_attn_mask
            )
            
            # Calculate loss
            loss = self.criterion(output.contiguous().view(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1))
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
            
            # Update parameters
            self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step()
            self.learning_rates.append(self.scheduler.get_lr())
            
            # Update total loss
            total_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % self.config['log_interval'] == 0:
                elapsed = time.time() - start_time
                print(f"Batch {batch_idx + 1}/{len(train_dataloader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"LR: {self.scheduler.get_lr():.7f} | "
                      f"Time: {elapsed:.2f}s")
                start_time = time.time()
                
        return total_loss / len(train_dataloader)
        
    def evaluate(self, val_dataloader: DataLoader) -> float:
        """Evaluate the model on validation data."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                
                # Create masks
                src_mask = self.model.create_pad_mask(src, self.src_tokenizer.pad_token_id).to(self.device)
                tgt_mask = self.model.create_pad_mask(tgt[:, :-1], self.tgt_tokenizer.pad_token_id).to(self.device)
                tgt_attn_mask = self.model.generate_square_subsequent_mask(tgt.size(1) - 1).to(self.device)
                
                # Forward pass
                output = self.model(
                    src=src,
                    tgt=tgt[:, :-1],
                    src_mask=src_mask,
                    tgt_mask=tgt_mask & tgt_attn_mask
                )
                
                # Calculate loss
                loss = self.criterion(output.contiguous().view(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1))
                
                # Update total loss
                total_loss += loss.item()
                
        return total_loss / len(val_dataloader)
    
    def train(
        self, 
        train_dataloader: DataLoader, 
        val_dataloader: DataLoader, 
        epochs: int
    ) -> Dict:
        """Train the model for multiple epochs."""
        print("Starting training...")
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_dataloader)
            self.train_losses.append(train_loss)
            
            # Evaluate
            val_loss = self.evaluate(val_dataloader)
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Save checkpoint if validation loss improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
                print(f"New best validation loss: {val_loss:.4f}, checkpoint saved.")
                
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates
        }
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler._step,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }, filename)
        
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler._step = checkpoint['scheduler_state']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
    def plot_training_history(self):
        """Plot training and validation loss history."""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.learning_rates)
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
