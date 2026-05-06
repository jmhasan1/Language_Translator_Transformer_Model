# Language Translator Transformer Model

A neural machine translation system built with PyTorch, implementing the Transformer architecture from the ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) paper.

## Project Overview

This project implements a language translation system using the Transformer architecture. It includes data preprocessing, model training, evaluation, inference, and a web interface for easy translation.

### Features

- Full implementation of the Transformer model with multi-head attention
- Custom tokenizer based on SentencePiece/BPE principles
- Training pipeline with configurable hyperparameters
- Beam search decoding for improved translation quality
- Web application for easy translation usage
- Model export options for deployment (TorchScript, ONNX)

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.7+
- CUDA-capable GPU recommended (but can run on CPU)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/jmhasan1/Language_Translator_Transformer_Model.git
cd Language_Translator_Transformer_Model
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Project Structure

```
Language_Translator_Transformer/
├── configs/                     # Configuration files
│   ├── model_config.py          # Model hyperparameters
│   └── training_config.py       # Training settings
├── data/
│   ├── preprocessing/           # Data preparation scripts
│   │   ├── tokenizer.py
│   │   └── dataset_utils.py
│   └── datasets/                # Raw and processed datasets
├── model/
│   ├── layers/                  # Individual model components
│   │   ├── attention.py
│   │   ├── embedding.py
│   │   ├── encoder.py
│   │   └── decoder.py
│   ├── transformer.py           # Main model architecture
│   └── model_utils.py           # Helper functions
├── training/
│   ├── trainer.py               # Training loop and logic
│   └── evaluation.py            # Evaluation metrics
├── inference/
│   ├── translate.py             # Translation functionality
│   └── beam_search.py           # Advanced decoding
├── webapp/
│   ├── app.py                   # Flask/Streamlit application
│   ├── static/                  # CSS, JS files
│   └── templates/               # HTML templates
├── scripts/
│   ├── train.py                 # Training entry point
│   ├── evaluate.py              # Evaluation script
│   └── export_model.py          # Export for deployment
├── requirements.txt
├── README.md
└── .gitignore
```

## Usage

### Data Preparation

1. Place your parallel corpus in the `data/datasets/raw/` directory.
2. Process the raw dataset:
```bash
python -m data.preprocessing.dataset_utils --src-lang en --tgt-lang fr --data-dir data/datasets/raw/ --output-dir data/datasets/processed/
```

3. Train the tokenizers:
```bash
python -m data.preprocessing.tokenizer --train --src-file data/datasets/processed/train.en --tgt-file data/datasets/processed/train.fr --vocab-size 16000
```

### Training

Train the model using the provided script:

```bash
python scripts/train.py --data-dir data/datasets/processed/ --src-lang en --tgt-lang fr --batch-size 32 --epochs 10 --save-dir checkpoints/
```

You can customize training by modifying the configuration files in the `configs/` directory.

### Evaluation

Evaluate model performance on test data:

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --test-src data/datasets/processed/test.en --test-tgt data/datasets/processed/test.fr
```

### Translation

Translate text using a trained model:

```bash
python inference/translate.py --text "Hello, how are you?" --src-lang en --tgt-lang fr --model-path checkpoints/best_model.pt
```

### Model Export

Export the trained model for deployment:

```bash
python scripts/export_model.py --checkpoint checkpoints/best_model.pt --src-tokenizer checkpoints/tokenizer_src.json --tgt-tokenizer checkpoints/tokenizer_tgt.json --export-dir exported_model/ --format all
```

### Web Application

Run the web application locally:

```bash
python webapp/app.py
```

Then navigate to `http://localhost:5000` in your web browser.

## Performance Optimization for Limited Hardware

This project includes several optimizations to run efficiently on systems with limited resources:

1. **Gradient accumulation** to simulate larger batch sizes
2. **Mixed precision training** using PyTorch AMP
3. **Memory-efficient attention mechanisms**
4. **Model checkpointing** to save the best performing models

You can adjust the configuration files to balance performance and resource usage according to your hardware capabilities.

## License

[MIT License](LICENSE)

## Acknowledgements

- ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) paper by Google and University of Torronto, Canada
- PyTorch documentation and tutorials
- The open-source NLP community

# Jahid MD HASAN