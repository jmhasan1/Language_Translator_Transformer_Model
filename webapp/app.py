from flask import Flask, render_template, request, jsonify
import torch
import os
import json
from typing import Dict, List, Optional

# Import our modules
from beam_search import BeamSearchTranslator
from improved_tokenization import TokenizerSentencePiece
from improved_transformer import Transformer


app = Flask(__name__)

# Configuration
MODEL_DIR = "models"
AVAILABLE_MODELS = []


def load_model_configs():
    """Load available model configurations."""
    global AVAILABLE_MODELS
    
    if os.path.exists(os.path.join(MODEL_DIR, "model_configs.json")):
        with open(os.path.join(MODEL_DIR, "model_configs.json"), "r") as f:
            AVAILABLE_MODELS = json.load(f)
    else:
        AVAILABLE_MODELS = []


def load_translator(model_name: str):
    """Load model and tokenizers for translation."""
    # Find model config
    model_config = None
    for config in AVAILABLE_MODELS:
        if config["name"] == model_name:
            model_config = config
            break
            
    if model_config is None:
        raise ValueError(f"Model {model_name} not found in available models.")
        
    # Load source tokenizer
    src_tokenizer = TokenizerSentencePiece()
    src_tokenizer.load(os.path.join(MODEL_DIR, model_config["src_tokenizer"]))
    
    # Load target tokenizer
    tgt_tokenizer = TokenizerSentencePiece()
    tgt_tokenizer.load(os.path.join(MODEL_DIR, model_config["tgt_tokenizer"]))
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Transformer(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        d_model=model_config["d_model"],
        num_encoder_layers=model_config["num_encoder_layers"],
        num_decoder_layers=model_config["num_decoder_layers"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],
        dropout=model_config["dropout"],
        pre_norm=model_config["pre_norm"]
    )
    
    # Load model weights
    model.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, model_config["weights"]),
            map_location=device
        )
    )
    
    # Initialize translator
    translator = BeamSearchTranslator(
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        device=device,
        beam_size=5
    )
    
    return translator, model_config


# Load available models on startup
load_model_configs()

# Store loaded translators
translators = {}


@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html", models=AVAILABLE_MODELS)


@app.route("/translate", methods=["POST"])
def translate():
    """Translate text from source to target language."""
    data = request.get_json()
    text = data.get("text", "")
    model_name = data.get("model", "")
    
    if not text:
        return jsonify({"error": "No text provided."}), 400
        
    if not model_name:
        return jsonify({"error": "No model selected."}), 400
        
    # Load translator if not already loaded
    if model_name not in translators:
        try:
            translator, config = load_translator(model_name)
            translators[model_name] = {
                "translator": translator,
                "config": config
            }
        except Exception as e:
            return jsonify({"error": f"Failed to load model: {str(e)}"}), 500
    
    # Translate text
    try:
        translation = translators[model_name]["translator"].translate(text)
        return jsonify({
            "translation": translation,
            "model": model_name,
            "source_lang": translators[model_name]["config"]["source_language"],
            "target_lang": translators[model_name]["config"]["target_language"]
        })
    except Exception as e:
        return jsonify({"error": f"Translation failed: {str(e)}"}), 500


@app.route("/models")
def list_models():
    """List available models."""
    return jsonify(AVAILABLE_MODELS)


if __name__ == "__main__":
    # Make sure the models directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Make sure static and templates directories exist
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)
