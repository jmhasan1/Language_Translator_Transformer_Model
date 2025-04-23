#!/usr/bin/env python3
"""
Script to export the trained transformer model for deployment.
Supports exporting to TorchScript, ONNX, or saving model weights.
"""

import os
import argparse
import torch
import logging
from pathlib import Path

from configs.model_config import ModelConfig
from model.transformer import Transformer
from data.preprocessing.tokenizer import Tokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path, model_config):
    """Load a trained model from checkpoint."""
    logger.info(f"Loading model from {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(model_config).to(device)
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device


def export_torchscript(model, export_dir, model_name="translator_model"):
    """Export the model to TorchScript format."""
    logger.info("Exporting to TorchScript format")
    
    script_model = torch.jit.script(model)
    
    # Create export directory if it doesn't exist
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, f"{model_name}.pt")
    
    # Save the model
    script_model.save(export_path)
    logger.info(f"TorchScript model saved to {export_path}")
    
    return export_path


def export_onnx(model, export_dir, model_config, device, model_name="translator_model"):
    """Export the model to ONNX format."""
    logger.info("Exporting to ONNX format")
    
    # Create dummy inputs for ONNX export
    src_seq_len = model_config.max_seq_length
    tgt_seq_len = model_config.max_seq_length
    batch_size = 1
    
    dummy_src = torch.randint(0, model_config.src_vocab_size, (batch_size, src_seq_len)).to(device)
    dummy_tgt = torch.randint(0, model_config.tgt_vocab_size, (batch_size, tgt_seq_len)).to(device)
    
    # Create export directory if it doesn't exist
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, f"{model_name}.onnx")
    
    # Export to ONNX
    torch.onnx.export(
        model,                      # model being run
        (dummy_src, dummy_tgt),     # model input (tuple)
        export_path,                # where to save the model
        export_params=True,         # store the trained parameter weights
        opset_version=12,           # the ONNX version to export to
        do_constant_folding=True,   # optimize constants
        input_names=['src', 'tgt'], # the model's input names
        output_names=['output'],    # the model's output names
        dynamic_axes={
            'src': {0: 'batch_size', 1: 'src_seq_len'},
            'tgt': {0: 'batch_size', 1: 'tgt_seq_len'},
            'output': {0: 'batch_size', 1: 'tgt_seq_len', 2: 'vocab_size'}
        }
    )
    
    logger.info(f"ONNX model saved to {export_path}")
    return export_path


def export_weights(model, export_dir, model_name="translator_model"):
    """Export only the model weights for easy loading."""
    logger.info("Exporting model weights")
    
    # Create export directory if it doesn't exist
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, f"{model_name}_weights.pt")
    
    # Save just the weights
    torch.save(model.state_dict(), export_path)
    logger.info(f"Model weights saved to {export_path}")
    
    return export_path


def export_tokenizers(src_tokenizer, tgt_tokenizer, export_dir):
    """Export tokenizers for the model."""
    logger.info("Exporting tokenizers")
    
    # Create export directory if it doesn't exist
    os.makedirs(export_dir, exist_ok=True)
    
    # Save tokenizers
    src_path = os.path.join(export_dir, "src_tokenizer.json")
    src_tokenizer.save(src_path)
    
    tgt_path = os.path.join(export_dir, "tgt_tokenizer.json")
    tgt_tokenizer.save(tgt_path)
    
    logger.info(f"Tokenizers saved to {export_dir}")
    
    return src_path, tgt_path


def export_metadata(model_config, export_dir):
    """Export model metadata and configuration."""
    logger.info("Exporting model metadata")
    
    # Create export directory if it doesn't exist
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, "model_config.json")
    
    # Save configuration
    model_config.save(export_path)
    logger.info(f"Model configuration saved to {export_path}")
    
    return export_path


def main():
    parser = argparse.ArgumentParser(description="Export trained model for deployment")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--src-tokenizer", required=True, help="Path to source tokenizer")
    parser.add_argument("--tgt-tokenizer", required=True, help="Path to target tokenizer")
    parser.add_argument("--export-dir", default="exported_model", help="Directory to save exported model")
    parser.add_argument("--format", choices=["torchscript", "onnx", "weights", "all"], 
                        default="all", help="Export format")
    parser.add_argument("--model-name", default="translator_model", help="Name for the exported model")
    
    args = parser.parse_args()
    
    # Load model configuration
    model_config = ModelConfig.from_json(os.path.join(os.path.dirname(args.checkpoint), "model_config.json"))
    
    # Load model
    model, device = load_model(args.checkpoint, model_config)
    
    # Load tokenizers
    src_tokenizer = Tokenizer.load(args.src_tokenizer)
    tgt_tokenizer = Tokenizer.load(args.tgt_tokenizer)
    
    # Update vocab sizes in model config if needed
    model_config.src_vocab_size = len(src_tokenizer)
    model_config.tgt_vocab_size = len(tgt_tokenizer)
    
    # Create export directory
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Export model based on format
    if args.format in ["torchscript", "all"]:
        export_torchscript(model, export_dir, args.model_name)
    
    if args.format in ["onnx", "all"]:
        export_onnx(model, export_dir, model_config, device, args.model_name)
    
    if args.format in ["weights", "all"]:
        export_weights(model, export_dir, args.model_name)
    
    # Export tokenizers and metadata
    export_tokenizers(src_tokenizer, tgt_tokenizer, export_dir)
    export_metadata(model_config, export_dir)
    
    logger.info(f"Model export complete. Files saved to {export_dir}")


if __name__ == "__main__":
    main()
