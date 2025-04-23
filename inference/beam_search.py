import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

class BeamSearchDecoder:
    """
    Implements beam search decoding for transformer-based translation models.
    
    Beam search maintains multiple translation hypotheses and expands each,
    keeping only the most likely ones at each step.
    """
    
    def __init__(self, model, tokenizer, max_length=50, beam_size=5, 
                 length_penalty=0.6, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the beam search decoder.
        
        Args:
            model: The transformer model for translation
            tokenizer: Tokenizer for source/target languages
            max_length: Maximum length of generated sequence
            beam_size: Number of beams to maintain
            length_penalty: Penalty factor for longer sequences
            device: Device to run computations on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.device = device
        
    def _length_penalty_score(self, length: int) -> float:
        """
        Calculate length penalty score based on sequence length.
        
        Args:
            length: Length of the sequence
            
        Returns:
            The length penalty score
        """
        return ((5 + length) / 6) ** self.length_penalty
    
    def decode(self, src_sentence: str) -> List[str]:
        """
        Perform beam search decoding to translate a source sentence.
        
        Args:
            src_sentence: Source language sentence
            
        Returns:
            List of decoded translations, sorted by score
        """
        # Tokenize and prepare source
        src_tokens = self.tokenizer.tokenize_source(src_sentence)
        src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(self.device)
        
        # Encode source sequence
        encoder_output = self.model.encode(src_tensor)
        
        # Initialize with start token
        start_token = self.tokenizer.get_target_start_token_id()
        hypotheses = torch.full((1, 1), start_token, dtype=torch.long).to(self.device)
        
        # Track scores for each hypothesis
        scores = torch.zeros(1, dtype=torch.float).to(self.device)
        
        # Store completed sequences and their scores
        finished_hypotheses = []
        finished_scores = []
        
        # Begin beam search
        for step in range(self.max_length):
            # Stop if all beams have finished
            if len(finished_hypotheses) >= self.beam_size:
                break
                
            # Get number of current hypotheses
            num_hypotheses = hypotheses.size(0)
            
            # Expand encoder outputs for each hypothesis
            expanded_encoder_output = encoder_output.expand(num_hypotheses, -1, -1)
            
            # Decode current hypotheses one step
            logits = self.model.decode(hypotheses, expanded_encoder_output)
            
            # Get probabilities for next tokens
            next_token_logits = logits[:, -1, :]
            
            # Apply softmax to convert to probabilities
            probs = F.log_softmax(next_token_logits, dim=-1)
            
            # Calculate scores for all possible next tokens
            if step == 0:
                # For first step, consider only the first hypothesis
                next_scores = probs[0].unsqueeze(0)
            else:
                # Add previous scores to current token scores
                next_scores = probs + scores.unsqueeze(1)
            
            # Flatten scores to find top k
            vocab_size = probs.size(-1)
            flat_scores = next_scores.view(-1)
            
            # Get top-k scores and corresponding indices
            top_k_scores, top_k_indices = flat_scores.topk(self.beam_size, sorted=True)
            
            # Convert flat indices to beam and token indices
            beam_indices = top_k_indices // vocab_size
            token_indices = top_k_indices % vocab_size
            
            # Create new hypotheses
            new_hypotheses = []
            new_scores = []
            
            # Process each beam
            for i, (beam_idx, token_idx, score) in enumerate(zip(beam_indices, token_indices, top_k_scores)):
                # Skip if we already have enough finished hypotheses
                if len(finished_hypotheses) >= self.beam_size and score < finished_scores[-1]:
                    continue
                
                # Get previous hypothesis and append new token
                prev_hypothesis = hypotheses[beam_idx]
                new_hypothesis = torch.cat([prev_hypothesis, token_idx.unsqueeze(0)], dim=0)
                
                # Check if sequence has ended
                if token_idx == self.tokenizer.get_target_end_token_id():
                    # Apply length penalty and add to finished hypotheses
                    normalized_score = score / self._length_penalty_score(len(new_hypothesis))
                    
                    if len(finished_hypotheses) < self.beam_size or normalized_score > finished_scores[-1]:
                        finished_hypotheses.append(new_hypothesis)
                        finished_scores.append(normalized_score)
                        
                        # Sort finished hypotheses by score
                        if len(finished_hypotheses) > self.beam_size:
                            # Remove lowest scoring hypothesis
                            idx = np.argmin(finished_scores)
                            finished_hypotheses.pop(idx)
                            finished_scores.pop(idx)
                else:
                    # Add to active hypotheses for next step
                    new_hypotheses.append(new_hypothesis)
                    new_scores.append(score)
                
                # Stop if we have enough new hypotheses
                if len(new_hypotheses) >= self.beam_size:
                    break
            
            # If no new hypotheses, all beams have ended
            if len(new_hypotheses) == 0:
                break
            
            # Stack new hypotheses and update scores
            hypotheses = torch.stack(new_hypotheses)
            scores = torch.tensor(new_scores, dtype=torch.float).to(self.device)
        
        # If we don't have enough finished hypotheses, add current active ones
        while len(finished_hypotheses) < self.beam_size and hypotheses.size(0) > 0:
            # Apply length penalty
            normalized_scores = scores / self._length_penalty_score(hypotheses.size(1))
            
            # Get best active hypothesis
            best_idx = normalized_scores.argmax()
            finished_hypotheses.append(hypotheses[best_idx])
            finished_scores.append(normalized_scores[best_idx])
            
            # Remove added hypothesis
            hypotheses = torch.cat([hypotheses[:best_idx], hypotheses[best_idx+1:]], dim=0)
            scores = torch.cat([scores[:best_idx], scores[best_idx+1:]], dim=0)
        
        # Sort finished hypotheses by score
        sorted_indices = np.argsort(finished_scores)[::-1]
        
        # Convert token IDs to words
        translations = []
        for idx in sorted_indices:
            tokens = finished_hypotheses[idx].cpu().numpy()
            # Remove start and end tokens
            tokens = tokens[1:-1] if tokens[-1] == self.tokenizer.get_target_end_token_id() else tokens[1:]
            translations.append(self.tokenizer.detokenize_target(tokens))
            
        return translations
    
    def greedy_decode(self, src_sentence: str) -> str:
        """
        Perform simple greedy decoding (faster but less accurate than beam search).
        
        Args:
            src_sentence: Source language sentence
            
        Returns:
            Translated sentence
        """
        # Tokenize source
        src_tokens = self.tokenizer.tokenize_source(src_sentence)
        src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(self.device)
        
        # Encode source
        encoder_output = self.model.encode(src_tensor)
        
        # Start with start token
        output = torch.tensor([[self.tokenizer.get_target_start_token_id()]], dtype=torch.long).to(self.device)
        
        # Generate tokens one by one
        for _ in range(self.max_length):
            # Get predictions
            logits = self.model.decode(output, encoder_output)
            
            # Get next token (most probable)
            next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            
            # Add to output
            output = torch.cat([output, next_token], dim=1)
            
            # Stop at end token
            if next_token.item() == self.tokenizer.get_target_end_token_id():
                break
        
        # Convert to sentence
        output_tokens = output[0, 1:].cpu().numpy()  # Skip start token
        # Remove end token if present
        if output_tokens[-1] == self.tokenizer.get_target_end_token_id():
            output_tokens = output_tokens[:-1]
            
        return self.tokenizer.detokenize_target(output_tokens)
