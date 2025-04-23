import torch
import sys
import os
import argparse
from typing import List, Optional, Union
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import TransformerModel
from data.preprocessing.tokenizer import Tokenizer
from inference.beam_search import BeamSearchDecoder

class Translator:
    """
    Main translator class for inference using a trained transformer model.
    """
    
    def __init__(self, model_path: str, tokenizer_src_path: str, tokenizer_tgt_path: str, 
                 device: str = None, max_length: int = 100):
        """
        Initialize the translator with model and tokenizers.
        
        Args:
            model_path: Path to the saved model checkpoint
            tokenizer_src_path: Path to the source language tokenizer
            tokenizer_tgt_path: Path to the target language tokenizer
            device: Device to run inference on ('cuda' or 'cpu')
            max_length: Maximum sequence length for translation
        """
        # Set device (use GPU if available unless specified otherwise)
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load tokenizers
        self.tokenizer = Tokenizer(
            tokenizer_src_path=tokenizer_src_path,
            tokenizer_tgt_path=tokenizer_tgt_path
        )
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()  # Set to evaluation mode
        
        # Create beam search decoder
        self.beam_decoder = BeamSearchDecoder(
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=max_length,
            device=self.device
        )
        
        self.max_length = max_length
        
    def _load_model(self, model_path: str) -> TransformerModel:
        """
        Load the transformer model from a saved checkpoint.
        
        Args:
            model_path: Path to the model checkpoint
            
        Returns:
            Loaded TransformerModel
        """
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model configuration
        config = checkpoint['config']
        
        # Create model with the saved configuration
        model = TransformerModel(
            src_vocab_size=config['src_vocab_size'],
            tgt_vocab_size=config['tgt_vocab_size'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            d_ff=config['d_ff'],
            max_seq_length=config['max_seq_length'],
            dropout=config['dropout']
        ).to(self.device)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def translate(self, sentence: str, beam_size: int = 5) -> Union[str, List[str]]:
        """
        Translate a source sentence to the target language.
        
        Args:
            sentence: Source language sentence
            beam_size: Beam size for decoding (if > 1, uses beam search)
            
        Returns:
            Translated sentence(s) in target language
        """
        with torch.no_grad():  # Disable gradient calculation for inference
            if beam_size > 1:
                # Use beam search for better quality
                translations = self.beam_decoder.decode(sentence)
                return translations[0] if len(translations) > 0 else ""
            else:
                # Use greedy decoding (faster)
                return self.beam_decoder.greedy_decode(sentence)
    
    def translate_batch(self, sentences: List[str], beam_size: int = 5) -> List[str]:
        """
        Translate a batch of sentences.
        
        Args:
            sentences: List of source language sentences
            beam_size: Beam size for decoding
            
        Returns:
            List of translated sentences
        """
        return [self.translate(sentence, beam_size) for sentence in sentences]

    def get_attention_weights(self, sentence: str) -> dict:
        """
        Get attention weights for visualization.
        
        Args:
            sentence: Source language sentence
            
        Returns:
            Dictionary with attention weights from different layers
        """
        # Tokenize input
        src_tokens = self.tokenizer.tokenize_source(sentence)
        src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(self.device)
        
        # Start with start token for target
        tgt_tokens = [self.tokenizer.get_target_start_token_id()]
        
        attention_weights = {
            'encoder_self_attention': [],
            'decoder_self_attention': [],
            'encoder_decoder_attention': []
        }
        
        with torch.no_grad():
            # Forward pass with attention weight tracking
            self.model.store_attention_weights = True
            
            # Encode source
            encoder_output = self.model.encode(src_tensor)
            
            # Generate target tokens and collect attention weights
            for _ in range(self.max_length):
                tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(self.device)
                
                # Decode step
                output = self.model.decode(tgt_tensor, encoder_output)
                next_token = output[:, -1, :].argmax(dim=-1).item()
                
                # Add token to output
                tgt_tokens.append(next_token)
                
                # Collect attention weights
                if hasattr(self.model, 'attention_weights'):
                    for key, value in self.model.attention_weights.items():
                        if key in attention_weights:
                            attention_weights[key].append(value)
                
                # Stop at end token
                if next_token == self.tokenizer.get_target_end_token_id():
                    break
            
            # Turn off attention weight storage
            self.model.store_attention_weights = False
        
        return attention_weights


def main():
    """
    Command-line interface for translation.
    """
    parser = argparse.ArgumentParser(description='Translate text using a trained transformer model')
    parser.add_argument('--model', required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--src-tokenizer', required=True, help='Path to the source tokenizer')
    parser.add_argument('--tgt-tokenizer', required=True, help='Path to the target tokenizer')
    parser.add_argument('--text', help='Text to translate')
    parser.add_argument('--input-file', help='File containing text to translate')
    parser.add_argument('--output-file', help='File to write translation to')
    parser.add_argument('--beam-size', type=int, default=5, help='Beam size for decoding')
    parser.add_argument('--device', choices=['cuda', 'cpu'], help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Initialize translator
    translator = Translator(
        model_path=args.model,
        tokenizer_src_path=args.src_tokenizer,
        tokenizer_tgt_path=args.tgt_tokenizer,
        device=args.device
    )
    
    if args.text:
        # Translate a single sentence
        translation = translator.translate(args.text, beam_size=args.beam_size)
        print(f"Source: {args.text}")
        print(f"Translation: {translation}")
        
    elif args.input_file:
        # Translate from file
        with open(args.input_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        
        translations = translator.translate_batch(sentences, beam_size=args.beam_size)
        
        if args.output_file:
            # Write to output file
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for translation in translations:
                    f.write(f"{translation}\n")
        else:
            # Print to console
            for src, tgt in zip(sentences, translations):
                print(f"Source: {src}")
                print(f"Translation: {tgt}")
                print("-" * 50)
    
    else:
        print("Please provide either --text or --input-file")

if __name__ == "__main__":
    main()
