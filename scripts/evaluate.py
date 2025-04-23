import os
import sys
import argparse
import torch
import logging
import json
import pandas as pd
import numpy as np
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import TransformerModel
from data.preprocessing.tokenizer import Tokenizer
from data.preprocessing.dataset_utils import TranslationDataset, collate_batch
from training.evaluation import calculate_bleu, calculate_rouge
from inference.translate import Translator
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate the transformer translation model")
    
    # Model parameters
    parser.add_argument("--model", required=True, help="Path to the model checkpoint")
    parser.add_argument("--src-tokenizer", required=True, help="Path to source tokenizer")
    parser.add_argument("--tgt-tokenizer", required=True, help="Path to target tokenizer")
    
    # Data parameters
    parser.add_argument("--src-data", required=True, help="Path to source language test data file")
    parser.add_argument("--tgt-data", required=True, help="Path to target language test data file (reference)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--max-seq-length", type=int, default=100, help="Maximum sequence length")
    
    # Translation parameters
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding")
    parser.add_argument("--output", help="Path to save translation outputs")
    
    # Hardware parameters
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Device to evaluate on (defaults to cuda if available)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    
    return parser.parse_args()

def evaluate_translations(references, hypotheses):
    """
    Calculate evaluation metrics for translations
    
    Args:
        references: List of reference translations (ground truth)
        hypotheses: List of model-generated translations
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Calculate BLEU score
    bleu = calculate_bleu(references, hypotheses)
    
    # Calculate ROUGE scores
    rouge_scores = calculate_rouge(references, hypotheses)
    
    # Return all metrics
    return {
        "bleu": bleu,
        "rouge1_f": rouge_scores["rouge1_f"],
        "rouge2_f": rouge_scores["rouge2_f"],
        "rougeL_f": rouge_scores["rougeL_f"]
    }

def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    
    # Load tokenizers
    logger.info("Loading tokenizers...")
    tokenizer = Tokenizer(
        tokenizer_src_path=args.src_tokenizer,
        tokenizer_tgt_path=args.tgt_tokenizer
    )
    
    # Initialize translator
    logger.info("Loading model and initializing translator...")
    translator = Translator(
        model_path=args.model,
        tokenizer_src_path=args.src_tokenizer,
        tokenizer_tgt_path=args.tgt_tokenizer,
        device=device.type,
        max_length=args.max_seq_length
    )
    
    # Load test data
    logger.info("Loading test data...")
    with open(args.src_data, 'r', encoding='utf-8') as f:
        src_sentences = [line.strip() for line in f if line.strip()]
    
    with open(args.tgt_data, 'r', encoding='utf-8') as f:
        tgt_sentences = [line.strip() for line in f if line.strip()]
    
    # Verify data alignmentIf some preprocessing step caused misalignment
    min_len = min(len(src_sentences), len(tgt_sentences))
    if len(src_sentences) != len(tgt_sentences):
        logger.warning(f"Source and target files have different number of lines. "
                      f"Using the first {min_len} lines from each.")
        src_sentences = src_sentences[:min_len]
        tgt_sentences = tgt_sentences[:min_len]
    
    # Translate source sentences
    logger.info(f"Translating {len(src_sentences)} sentences with beam size {args.beam_size}...")
    start_time = time.time()
    
    # Process in batches to show progress
    batch_size = 100  # For progress reporting
    translations = []
    
    for i in range(0, len(src_sentences), batch_size):
        batch = src_sentences[i:i+batch_size]
        batch_translations = translator.translate_batch(batch, beam_size=args.beam_size)
        translations.extend(batch_translations)
        
        # Report progress
        progress = min(i + batch_size, len(src_sentences))
        logger.info(f"Translated {progress}/{len(src_sentences)} sentences "
                   f"({progress/len(src_sentences)*100:.1f}%)")
    
    translation_time = time.time() - start_time
    logger.info(f"Translation completed in {translation_time:.2f} seconds "
               f"({len(src_sentences)/translation_time:.2f} sentences/sec)")
    
    # Calculate evaluation metrics
    logger.info("Calculating evaluation metrics...")
    metrics = evaluate_translations(tgt_sentences, translations)
    
    # Print metrics
    logger.info("Evaluation Results:")
    logger.info(f"BLEU score: {metrics['bleu']:.4f}")
    logger.info(f"ROUGE-1 F1: {metrics['rouge1_f']:.4f}")
    logger.info(f"ROUGE-2 F1: {metrics['rouge2_f']:.4f}")
    logger.info(f"ROUGE-L F1: {metrics['rougeL_f']:.4f}")
    
    # Save results if output path specified
    if args.output:
        # Create DataFrame with source, reference, and translation
        results_df = pd.DataFrame({
            "source": src_sentences,
            "reference": tgt_sentences,
            "translation": translations
        })
        
        # Save to CSV
        results_df.to_csv(args.output, index=False)
        logger.info(f"Translation results saved to {args.output}")
        
        # Save metrics to JSON
        metrics_path = os.path.splitext(args.output)[0] + "_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Evaluation metrics saved to {metrics_path}")
    
    return metrics

if __name__ == "__main__":
    main()
