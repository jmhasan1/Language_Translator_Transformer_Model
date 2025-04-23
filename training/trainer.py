import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import logging
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union, Callable

from model.transformer import Transformer
from model.model_utils import save_model, create_masks, LabelSmoothingLoss, NoamScheduler
from training.evaluation import evaluate_bleu
from configs.training_config import TrainingConfig


class Trainer:
    """
    Trainer class for managing the training process of the transformer model.
    """
    
    def __init__(
        self,
        model: Transformer,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        config: TrainingConfig,
        device: torch.device,
        tokenizer_src=None,
        tokenizer_tgt=None,
        checkpoint_dir: str = "checkpoints"
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Transformer model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            config: Training configuration
            device: Device to train on
            tokenizer_src: Source language tokenizer
            tokenizer_tgt: Target language tokenizer
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.checkpoint_dir = checkpoint_dir
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("Trainer")
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Setup criterion
        if config.label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(
                smoothing=config.label_smoothing,
                ignore_index=config.pad_idx
            )
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
            weight_decay=config.weight_decay
        )
        
        # Setup learning rate scheduler
        if config.use_noam_scheduler:
            self.scheduler = NoamScheduler(
                optimizer=self.optimizer,
                d_model=model.d_model,
                warmup_steps=config.warmup_steps,
                factor=config.lr_factor
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=config.lr_patience,
                verbose=True
            )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_bleu = 0.0
        self.training_losses = []
        self.validation_losses = []
        self.bleu_scores = []
        
    def train(self, epochs: int) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.
        
        Args:
            epochs: Number of epochs to train
            
        Returns:
            Dictionary of training history
        """
        self.logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            # Train for one epoch
            train_loss = self._train_epoch(epoch)
            self.training_losses.append(train_loss)
            
            # Validate
            val_loss, bleu = self._validate_epoch(epoch)
            self.validation_losses.append(val_loss)
            self.bleu_scores.append(bleu)
            
            # Update learning rate if using ReduceLROnPlateau
            if not self.config.use_noam_scheduler:
                self.scheduler.step(val_loss)
            
            # Save checkpoint
            self._save_checkpoint(epoch, val_loss, bleu)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, bleu, is_best=True)
                
            if bleu > self.best_bleu:
                self.best_bleu = bleu
                self._save_checkpoint(epoch, val_loss, bleu, is_best_bleu=True)
                
            self.current_epoch = epoch + 1
            
        history = {
            'train_loss': self.training_losses,
            'val_loss': self.validation_losses,
            'bleu': self.bleu_scores
        }
        
        return history
    
    def _train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        epoch_loss = 0
        start_time = time.time()
        
        # Setup progress bar
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch+1} [Training]",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            
            # During training, target input is target without the last token
            # Target output is target without the first token (shift right)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Create masks
            src_mask, tgt_mask, memory_mask = create_masks(
                src, tgt_input, self.config.pad_idx
            )
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # For very large models, use gradient checkpointing to save memory
            if self.config.use_gradient_checkpointing:
                self.model.encoder.transformer_encoder.config.gradient_checkpointing = True
                self.model.decoder.transformer_decoder.config.gradient_checkpointing = True
            
            # Get model output
            output = self.model(
                src=src,
                tgt=tgt_input,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask
            )
            
            # Calculate loss
            # Reshape output and target for loss calculation
            # output shape: [batch_size, seq_len, vocab_size]
            # tgt_output shape: [batch_size, seq_len]
            output_flat = output.contiguous().view(-1, output.size(-1))
            tgt_output_flat = tgt_output.contiguous().view(-1)
            
            loss = self.criterion(output_flat, tgt_output_flat)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.clip_grad_norm
                )
            
            # Update weights
            self.optimizer.step()
            
            # Update learning rate if using Noam scheduler
            if self.config.use_noam_scheduler:
                self.scheduler.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Free up memory
            del src, tgt, tgt_input, tgt_output, src_mask, tgt_mask, memory_mask, output
            torch.cuda.empty_cache()
            
            # Update progress bar
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.7f}"
            )
            
            # Accumulate the global step
            self.global_step += 1
            
            # Add logging during training
            if batch_idx % self.config.log_interval == 0:
                self.logger.info(
                    f"Epoch: {epoch+1}, Batch: {batch_idx}/{len(self.train_dataloader)}, "
                    f"Loss: {loss.item():.4f}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.7f}"
                )
                
        # Calculate average loss
        avg_loss = epoch_loss / len(self.train_dataloader)
        
        # Log epoch statistics
        elapsed_time = time.time() - start_time
        self.logger.info(
            f"Epoch {epoch+1} completed in {elapsed_time:.2f}s | "
            f"Training Loss: {avg_loss:.4f}"
        )
        
        return avg_loss
    
    def _validate_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Validate the model on the validation set.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (validation loss, BLEU score)
        """
        if self.val_dataloader is None:
            return 0.0, 0.0
            
        self.model.eval()
        val_loss = 0
        start_time = time.time()
        
        # Setup progress bar
        progress_bar = tqdm(
            self.val_dataloader,
            desc=f"Epoch {epoch+1} [Validation]",
            leave=False
        )
        
        # Lists to store predictions and references for BLEU calculation
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            for batch in progress_bar:
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                
                # During validation, target input is target without the last token
                # Target output is target without the first token (shift right)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                # Create masks
                src_mask, tgt_mask, memory_mask = create_masks(
                    src, tgt_input, self.config.pad_idx
                )
                
                # Forward pass
                output = self.model(
                    src=src,
                    tgt=tgt_input,
                    src_mask=src_mask,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask
                )
                
                # Calculate loss
                output_flat = output.contiguous().view(-1, output.size(-1))
                tgt_output_flat = tgt_output.contiguous().view(-1)
                
                loss = self.criterion(output_flat, tgt_output_flat)
                val_loss += loss.item()
                
                # Get predictions for BLEU score calculation
                # We only need a subset of validation batches to calculate BLEU
                if len(all_predictions) < self.config.bleu_max_samples:
                    # Get predictions
                    _, predictions = output.max(dim=-1)
                    
                    # Convert tensors to lists
                    for i in range(min(predictions.size(0), self.config.bleu_max_samples - len(all_predictions))):
                        # Skip if already have enough samples
                        if len(all_predictions) >= self.config.bleu_max_samples:
                            break
                            
                        pred_tokens = predictions[i].tolist()
                        tgt_tokens = tgt_output[i].tolist()
                        
                        # Remove padding tokens
                        pred_tokens = [t for t in pred_tokens if t != self.config.pad_idx]
                        tgt_tokens = [t for t in tgt_tokens if t != self.config.pad_idx]
                        
                        # Convert token IDs to words if tokenizer is available
                        if self.tokenizer_tgt is not None:
                            pred_text = self.tokenizer_tgt.decode(pred_tokens)
                            tgt_text = self.tokenizer_tgt.decode(tgt_tokens)
                            all_predictions.append(pred_text)
                            all_references.append([tgt_text])
                        else:
                            all_predictions.append(pred_tokens)
                            all_references.append([tgt_tokens])
                
                # Update progress bar
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                
                # Free up memory
                del src, tgt, tgt_input, tgt_output, output
                torch.cuda.empty_cache()
        
        # Calculate average loss
        avg_loss = val_loss / len(self.val_dataloader)
        
        # Calculate BLEU score if predictions are available
        bleu_score = 0.0
        if all_predictions and all_references:
            bleu_score = evaluate_bleu(all_predictions, all_references)
        
        # Log validation statistics
        elapsed_time = time.time() - start_time
        self.logger.info(
            f"Epoch {epoch+1} validation completed in {elapsed_time:.2f}s | "
            f"Validation Loss: {avg_loss:.4f} | BLEU: {bleu_score:.4f}"
        )
        
        return avg_loss, bleu_score
    
    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        bleu: float,
        is_best: bool = False,
        is_best_bleu: bool = False
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            bleu: BLEU score
            is_best: Whether this is the best model by loss
            is_best_bleu: Whether this is the best model by BLEU
        """
        # Create checkpoint dictionary
        config_dict = self.config.__dict__
        
        # Add paths to tokenizers if available
        tokenizer_src_path = None
        tokenizer_tgt_path = None
        
        if hasattr(self.tokenizer_src, 'save_path'):
            tokenizer_src_path = self.tokenizer_src.save_path
            
        if hasattr(self.tokenizer_tgt, 'save_path'):
            tokenizer_tgt_path = self.tokenizer_tgt.save_path
        
        # Save regular checkpoint
        save_model(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=val_loss,
            save_dir=self.checkpoint_dir,
            config=config_dict,
            tokenizer_src_path=tokenizer_src_path,
            tokenizer_tgt_path=tokenizer_tgt_path,
            name=f"epoch_{epoch}"
        )
        
        # Save best model by loss
        if is_best:
            save_model(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                loss=val_loss,
                save_dir=self.checkpoint_dir,
                config=config_dict,
                tokenizer_src_path=tokenizer_src_path,
                tokenizer_tgt_path=tokenizer_tgt_path,
                name="best_model"
            )
            self.logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # Save best model by BLEU
        if is_best_bleu:
            save_model(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                loss=val_loss,
                save_dir=self.checkpoint_dir,
                config=config_dict,
                tokenizer_src_path=tokenizer_src_path,
                tokenizer_tgt_path=tokenizer_tgt_path,
                name="best_bleu_model"
            )
            self.logger.info(f"Saved best BLEU model with score: {bleu:.4f}")