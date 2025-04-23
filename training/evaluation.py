import torch
import numpy as np
from typing import List, Dict, Tuple, Union, Optional
from collections import Counter
import math
import re
import time
import logging
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)


def evaluate_bleu(
    predictions: List[Union[str, List[int]]],
    references: List[List[Union[str, List[int]]]],
    max_n: int = 4,
    weights: Optional[List[float]] = None
) -> float:
    """
    Calculate BLEU score for translation evaluation.
    Implemented based on the original BLEU paper: Papineni et al. (2002)
    
    Args:
        predictions: List of predicted sentences (strings or token IDs)
        references: List of reference sentences (list of lists: multiple references per sentence)
        max_n: Maximum n-gram size
        weights: Weights for different n-gram precisions (default: uniform weights)
        
    Returns:
        BLEU score (between 0 and 1)
    """
    # Default weights if not provided
    if weights is None:
        weights = [1.0 / max_n] * max_n
    
    # Initialize counters
    prediction_len = 0
    reference_len = 0
    
    # Modified n-gram precision scores
    precisions = [0.0] * max_n
    
    # Calculate precision for each n-gram size
    for pred, refs in zip(predictions, references):
        # Convert strings to lists of tokens if needed
        if isinstance(pred, str):
            pred_tokens = pred.split()
        else:
            pred_tokens = pred
            
        ref_tokens_list = []
        for ref in refs:
            if isinstance(ref, str):
                ref_tokens_list.append(ref.split())
            else:
                ref_tokens_list.append(ref)
        
        # Update lengths
        prediction_len += len(pred_tokens)
        
        # Find closest reference length
        ref_lens = [len(ref) for ref in ref_tokens_list]
        ref_len_closest = min(ref_lens, key=lambda x: abs(x - len(pred_tokens)))
        reference_len += ref_len_closest
        
        # Calculate n-gram matches for each n
        for n in range(1, max_n + 1):
            matches, total = _get_ngram_matches(pred_tokens, ref_tokens_list, n)
            if total > 0:
                precisions[n-1] += matches / total
    
    # Average precision scores across all sentences
    num_sentences = len(predictions)
    if num_sentences > 0:
        precisions = [p / num_sentences for p in precisions]
    
    # Calculate brevity penalty
    if prediction_len < reference_len:
        brevity_penalty = math.exp(1 - reference_len / prediction_len)
    else:
        brevity_penalty = 1.0
    
    # Calculate weighted sum of log precisions
    score = 0.0
    for i, precision in enumerate(precisions):
        if precision > 0:
            score += weights[i] * math.log(precision)
    
    # Apply brevity penalty and convert to score between 0 and 1
    bleu = brevity_penalty * math.exp(score)
    
    return bleu


def _get_ngram_matches(
    pred_tokens: List[Union[str, int]],
    ref_tokens_list: List[List[Union[str, int]]],
    n: int
) -> Tuple[int, int]:
    """
    Count n-gram matches between prediction and references.
    
    Args:
        pred_tokens: List of predicted tokens
        ref_tokens_list: List of reference token lists
        n: n-gram size to consider
        
    Returns:
        Tuple of (number of matched n-grams, total number of n-grams in prediction)
    """
    # Extract n-grams from prediction
    pred_ngrams = _extract_ngrams(pred_tokens, n)
    pred_ngram_counts = Counter(pred_ngrams)
    
    # No n-grams in prediction
    if not pred_ngram_counts:
        return 0, 0
    
    # Extract n-grams from references
    ref_ngram_counts_list = [Counter(_extract_ngrams(ref_tokens, n)) for ref_tokens in ref_tokens_list]
    
    # Find maximum matches
    matches = 0
    for ngram, count in pred_ngram_counts.items():
        # Find max count of this n-gram in any reference
        max_count = max(ref_count.get(ngram, 0) for ref_count in ref_ngram_counts_list)
        # Add minimum of prediction count and max reference count
        matches += min(count, max_count)
    
    # Total number of n-grams in prediction
    total = sum(pred_ngram_counts.values())
    
    return matches, total


def _extract_ngrams(tokens: List[Union[str, int]], n: int) -> List[Tuple[Union[str, int], ...]]:
    """
    Extract n-grams from a list of tokens.
    
    Args:
        tokens: List of tokens
        n: n-gram size
        
    Returns:
        List of n-gram tuples
    """
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i+n]))
    return ngrams


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    tokenizer_tgt=None,
    max_samples: int = None,
    pad_idx: int = 0
) -> Tuple[float, float]:
    """
    Evaluate model on dataset.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        criterion: Loss function
        device: Device to run evaluation on
        tokenizer_tgt: Target tokenizer (for BLEU calculation)
        max_samples: Maximum number of samples to evaluate
        pad_idx: Padding token index
        
    Returns:
        Tuple of (loss, BLEU score)
    """
    model.eval()
    loss_total = 0
    start_time = time.time()
    
    # Lists to store predictions and references for BLEU calculation
    all_predictions = []
    all_references = []
    
    # Limit the number of batches if max_samples is specified
    num_batches = len(dataloader)
    if max_samples is not None:
        # Calculate how many batches needed based on batch size
        batch_size = next(iter(dataloader))['src'].size(0)
        num_batches = min(num_batches, math.ceil(max_samples / batch_size))
    
    # Create masks function (assuming it's similar to the one in the training code)
    def create_masks(src, tgt):
        src_mask = (src == pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
        
        seq_len = tgt.size(1)
        nopeak_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        nopeak_mask = nopeak_mask.to(tgt.device)
        
        tgt_mask = tgt_mask | nopeak_mask.unsqueeze(0)
        
        return src_mask, tgt_mask
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating", total=num_batches)):
            if i >= num_batches:
                break
                
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            # Target input is all but the last token
            tgt_input = tgt[:, :-1]
            # Target output is all but the first token (shifted right)
            tgt_output = tgt[:, 1:]
            
            # Create masks
            src_mask, tgt_mask = create_masks(src, tgt_input)
            
            # Forward pass
            output = model(src, tgt_input, src_mask, tgt_mask)
            
            # Calculate loss
            loss = criterion(
                output.contiguous().view(-1, output.size(-1)),
                tgt_output.contiguous().view(-1)
            )
            
            loss_total += loss.item()
            
            # Get predictions
            _, predictions = torch.max(output, dim=-1)
            
            # Convert predictions and targets to lists for BLEU calculation
            for j in range(predictions.size(0)):
                # Predicted tokens
                pred_tokens = predictions[j].cpu().tolist()
                # Target tokens
                tgt_tokens = tgt_output[j].cpu().tolist()
                
                # Remove padding tokens
                pred_tokens = [token for token in pred_tokens if token != pad_idx]
                tgt_tokens = [token for token in tgt_tokens if token != pad_idx]
                
                # Convert tokens to text if tokenizer is available
                if tokenizer_tgt is not None:
                    try:
                        pred_text = tokenizer_tgt.decode(pred_tokens)
                        tgt_text = tokenizer_tgt.decode(tgt_tokens)
                        all_predictions.append(pred_text)
                        all_references.append([tgt_text])
                    except Exception as e:
                        logger.warning(f"Error decoding tokens: {e}")
                        continue
                else:
                    all_predictions.append(pred_tokens)
                    all_references.append([tgt_tokens])
    
    # Calculate average loss
    avg_loss = loss_total / num_batches
    
    # Calculate BLEU score
    bleu_score = evaluate_bleu(all_predictions, all_references)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Evaluation completed in {elapsed_time:.2f}s | Loss: {avg_loss:.4f} | BLEU: {bleu_score:.4f}")
    
    return avg_loss, bleu_score


def generate_translations(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    tokenizer_src,
    tokenizer_tgt,
    device: torch.device,
    max_samples: int = 10,
    max_length: int = 100,
    beam_size: int = 1,
    start_token_id: int = None,
    end_token_id: int = None
) -> List[Dict[str, str]]:
    """
    Generate translations for a set of samples.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        tokenizer_src: Source tokenizer
        tokenizer_tgt: Target tokenizer
        device: Device to run inference on
        max_samples: Maximum number of samples to translate
        max_length: Maximum output sequence length
        beam_size: Beam size for beam search decoding
        start_token_id: Start token ID (will use tokenizer's value if None)
        end_token_id: End token ID (will use tokenizer's value if None)
        
    Returns:
        List of dictionaries containing source, reference, and translated text
    """
    model.eval()
    results = []
    count = 0
    
    # Set start and end token IDs if not provided
    if start_token_id is None and hasattr(tokenizer_tgt, 'bos_token_id'):
        start_token_id = tokenizer_tgt.bos_token_id
    if end_token_id is None and hasattr(tokenizer_tgt, 'eos_token_id'):
        end_token_id = tokenizer_tgt.eos_token_id
    
    # Fallback to common values if still None
    if start_token_id is None:
        start_token_id = 2  # Common value for BOS
    if end_token_id is None:
        end_token_id = 3  # Common value for EOS
    
    # Helper function for simple greedy decoding
    def greedy_decode(src, max_len=max_length):
        # Create source mask
        src_mask = (src == tokenizer_src.pad_token_id).unsqueeze(1).unsqueeze(2)
        
        # Encode source
        with torch.no_grad():
            enc_output = model.encoder(src, src_mask)
        
        # Start with a single BOS token
        dec_input = torch.ones(1, 1).fill_(start_token_id).long().to(device)
        
        for _ in range(max_len - 1):
            # Don't let it look at future tokens
            tgt_mask = torch.zeros(1, dec_input.size(1), dec_input.size(1)).bool().to(device)
            
            # Get next token prediction
            with torch.no_grad():
                out = model.decoder(dec_input, enc_output, tgt_mask, src_mask)
                out = model.output_linear(out)
                prob = out[:, -1]
                _, next_word = torch.max(prob, dim=1)
                
                # Add the next word to the input
                dec_input = torch.cat(
                    [dec_input, next_word.unsqueeze(0)], dim=1
                )
            
            # Stop if we've generated an EOS token
            if next_word.item() == end_token_id:
                break
        
        return dec_input.squeeze(0)
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            for i in range(src.size(0)):
                if count >= max_samples:
                    break
                
                # Get single source and target
                src_sample = src[i].unsqueeze(0)
                tgt_sample = tgt[i]
                
                # Decode
                if beam_size > 1:
                    # Placeholder for beam search - would be implemented separately
                    pred_tokens = greedy_decode(src_sample)
                else:
                    pred_tokens = greedy_decode(src_sample)
                
                # Convert token IDs to text
                src_tokens = src_sample.squeeze(0).cpu().tolist()
                pred_tokens = pred_tokens.cpu().tolist()
                tgt_tokens = tgt_sample.cpu().tolist()
                
                # Remove padding, BOS, and EOS tokens
                src_tokens = [t for t in src_tokens if t != tokenizer_src.pad_token_id]
                pred_tokens = [t for t in pred_tokens if t != tokenizer_tgt.pad_token_id]
                tgt_tokens = [t for t in tgt_tokens if t != tokenizer_tgt.pad_token_id]
                
                # Remove BOS and EOS in the target tokens if present
                if hasattr(tokenizer_tgt, 'bos_token_id') and tgt_tokens[0] == tokenizer_tgt.bos_token_id:
                    tgt_tokens = tgt_tokens[1:]
                if hasattr(tokenizer_tgt, 'eos_token_id') and tgt_tokens[-1] == tokenizer_tgt.eos_token_id:
                    tgt_tokens = tgt_tokens[:-1]
                
                # Remove BOS and EOS in the predicted tokens if present
                if len(pred_tokens) > 0 and pred_tokens[0] == start_token_id:
                    pred_tokens = pred_tokens[1:]
                if len(pred_tokens) > 0 and pred_tokens[-1] == end_token_id:
                    pred_tokens = pred_tokens[:-1]
                
                # Decode tokens to text
                src_text = tokenizer_src.decode(src_tokens)
                pred_text = tokenizer_tgt.decode(pred_tokens)
                tgt_text = tokenizer_tgt.decode(tgt_tokens)
                
                # Store result
                results.append({
                    'source': src_text,
                    'reference': tgt_text,
                    'translation': pred_text
                })
                
                count += 1
                
                if count >= max_samples:
                    break
            
            if count >= max_samples:
                break
    
    return results