import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np
import time


class BeamSearchTranslator:
    def __init__(
        self,
        model: torch.nn.Module,
        src_tokenizer,
        tgt_tokenizer,
        device: torch.device,
        max_length: int = 128,
        beam_size: int = 5,
        length_penalty: float = 1.0
    ):
        self.model = model
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.device = device
        self.max_length = max_length
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        
        # Set model to evaluation mode
        self.model.eval()
        
    def translate(self, text: str) -> str:
        """Translate text using beam search."""
        # Tokenize source text
        src_tokens = self.src_tokenizer.encode(text)
        src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(self.device)
        
        # Create source mask
        src_mask = self.model.create_pad_mask(
            src_tensor, self.src_tokenizer.pad_token_id
        ).to(self.device)
        
        with torch.no_grad():
            # Encode source sequence
            memory, _ = self.model.encode(src_tensor, src_mask)
            
            # Initialize decoder input with BOS token
            ys = torch.ones(1, 1).fill_(self.tgt_tokenizer.bos_token_id).long().to(self.device)
            
            # Initialize beam
            beams = [(ys, 0.0)]  # (sequence, score)
            completed_beams = []
            
            # Beam search
            for i in range(self.max_length - 1):
                candidates = []
                
                for seq, score in beams:
                    # If sequence ends with EOS, add to completed beams
                    if seq[0, -1].item() == self.tgt_tokenizer.eos_token_id:
                        completed_beams.append((seq, score))
                        continue
                        
                    # Generate target mask
                    tgt_mask = self.model.create_pad_mask(
                        seq, self.tgt_tokenizer.pad_token_id
                    ).to(self.device)
                    
                    tgt_attn_mask = self.model.generate_square_subsequent_mask(
                        seq.size(1)
                    ).to(self.device)
                    
                    # Decode one step
                    out, _, _ = self.model.decode(
                        seq, memory, tgt_mask & tgt_attn_mask
                    )
                    
                    # Get probability distribution for next token
                    prob = F.log_softmax(self.model.output_projection(out[:, -1]), dim=-1)
                    
                    # Get top-k tokens
                    top_k_probs, top_k_indices = prob.topk(self.beam_size)
                    
                    # Add candidates to list
                    for j in range(self.beam_size):
                        token = top_k_indices[0, j].unsqueeze(0).unsqueeze(0)
                        new_seq = torch.cat([seq, token], dim=1)
                        new_score = score + top_k_probs[0, j].item()
                        candidates.append((new_seq, new_score))
                
                # If we have enough completed beams, stop
                if len(completed_beams) >= self.beam_size:
                    break
                    
                # Sort candidates by score
                candidates.sort(key=lambda x: x[1], reverse=True)
                
                # Keep top beam_size candidates
                beams = candidates[:self.beam_size]
                
                # If all beams end with EOS, stop
                if all(beam[0][0, -1].item() == self.tgt_tokenizer.eos_token_id for beam in beams):
                    completed_beams.extend(beams)
                    break
            
            # Add remaining beams to completed beams
            completed_beams.extend(beams)
            
            # Apply length penalty
            def apply_length_penalty(seq, score):
                length = seq.size(1)
                return score / (length ** self.length_penalty)
            
            # Sort completed beams by score with length penalty
            completed_beams.sort(
                key=lambda x: apply_length_penalty(x[0], x[1]), reverse=True
            )
            
            # Get best sequence
            best_seq = completed_beams[0][0].squeeze()
            
            # Decode
            output_tokens = best_seq.tolist()
            translation = self.tgt_tokenizer.decode(output_tokens)
            
            return translation
            
    def translate_batch(
        self, 
        texts: List[str],
        show_progress: bool = True
    ) -> List[str]:
        """Translate a batch of texts."""
        translations = []
        start_time = time.time()
        
        for i, text in enumerate(texts):
            if show_progress and (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Translated {i + 1}/{len(texts)} texts in {elapsed:.2f}s")
                
            translation = self.translate(text)
            translations.append(translation)
            
        return translations
    
    def evaluate_bleu(self, src_texts: List[str], tgt_texts: List[str]) -> float:
        """Evaluate translations using BLEU score."""
        from sacrebleu import corpus_bleu
        
        # Translate source texts
        translations = self.translate_batch(src_texts)
        
        # Calculate BLEU score
        bleu = corpus_bleu(translations, [tgt_texts])
        
        return bleu.score


class GreedyTranslator:
    def __init__(
        self,
        model: torch.nn.Module,
        src_tokenizer,
        tgt_tokenizer,
        device: torch.device,
        max_length: int = 128
    ):
        self.model = model
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.device = device
        self.max_length = max_length
        
        # Set model to evaluation mode
        self.model.eval()
        
    def translate(self, text: str) -> str:
        """Translate text using greedy search."""
        # Tokenize source text
        src_tokens = self.src_tokenizer.encode(text)
        src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(self.device)
        
        # Create source mask
        src_mask = self.model.create_pad_mask(
            src_tensor, self.src_tokenizer.pad_token_id
        ).to(self.device)
        
        with torch.no_grad():
            # Encode source sequence
            memory, _ = self.model.encode(src_tensor, src_mask)
            
            # Initialize decoder input with BOS token
            ys = torch.ones(1, 1).fill_(self.tgt_tokenizer.bos_token_id).long().to(self.device)
            
            # Generate translation
            for i in range(self.max_length - 1):
                # Generate target mask
                tgt_mask = self.model.create_pad_mask(
                    ys, self.tgt_tokenizer.pad_token_id
                ).to(self.device)
                
                tgt_attn_mask = self.model.generate_square_subsequent_mask(
                    ys.size(1)
                ).to(self.device)
                
                # Decode one step
                out, _, _ = self.model.decode(
                    ys, memory, tgt_mask & tgt_attn_mask
                )
                
                # Get probability distribution for next token
                prob = self.model.output_projection(out[:, -1])
                
                # Get next token
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.item()
                
                # Add next token to sequence
                ys = torch.cat([
                    ys, torch.ones(1, 1).fill_(next_word).long().to(self.device)
                ], dim=1)
                
                # Stop if EOS token is generated
                if next_word == self.tgt_tokenizer.eos_token_id:
                    break
            
            # Decode
            output_tokens = ys.squeeze().tolist()
            translation = self.tgt_tokenizer.decode(output_tokens)
            
            return translation
    
    def translate_batch(
        self, 
        texts: List[str],
        show_progress: bool = True
    ) -> List[str]:
        """Translate a batch of texts."""
        translations = []
        start_time = time.time()
        
        for i, text in enumerate(texts):
            if show_progress and (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Translated {i + 1}/{len(texts)} texts in {elapsed:.2f}s")
                
            translation = self.translate(text)
            translations.append(translation)
            
        return translations
