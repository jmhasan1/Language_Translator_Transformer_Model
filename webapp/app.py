import streamlit as st
import sys
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.translate import Translator
from configs.model_config import ModelConfig

# Set page configuration
st.set_page_config(
    page_title="Neural Machine Translator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths (will be overridden by command line arguments)
DEFAULT_MODEL_PATH = os.path.join("saved_models", "transformer_model.pt")
DEFAULT_SRC_TOKENIZER_PATH = os.path.join("saved_models", "tokenizer_src.json")
DEFAULT_TGT_TOKENIZER_PATH = os.path.join("saved_models", "tokenizer_tgt.json")

# Function to initialize or get the translator
@st.cache_resource
def get_translator(model_path, src_tokenizer_path, tgt_tokenizer_path, device=None):
    """Initialize the translator and cache it"""
    try:
        return Translator(
            model_path=model_path,
            tokenizer_src_path=src_tokenizer_path,
            tokenizer_tgt_path=tgt_tokenizer_path,
            device=device
        )
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def plot_attention_weights(weights, src_tokens, tgt_tokens, layer_idx=0, head_idx=0):
    """Plot attention weights as a heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract weights for the specified layer and head
    if 'encoder_decoder_attention' in weights and weights['encoder_decoder_attention']:
        # Get the encoder-decoder attention weights
        attn_weights = weights['encoder_decoder_attention'][layer_idx][0, head_idx].cpu().numpy()
        
        # Create a heatmap
        im = ax.imshow(attn_weights, cmap='viridis')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(src_tokens)))
        ax.set_yticks(np.arange(len(tgt_tokens)))
        
        ax.set_xticklabels(src_tokens, rotation=45, ha='right')
        ax.set_yticklabels(tgt_tokens)
        
        # Loop over data dimensions and create text annotations
        for i in range(len(tgt_tokens)):
            for j in range(len(src_tokens)):
                text = ax.text(j, i, f"{attn_weights[i, j]:.2f}",
                              ha="center", va="center", color="w" if attn_weights[i, j] > 0.5 else "black")
        
        ax.set_title(f"Encoder-Decoder Attention (Layer {layer_idx+1}, Head {head_idx+1})")
        fig.tight_layout()
        
        return fig
    
    return None

def main():
    """Main Streamlit app function"""
    st.title("Neural Machine Translator 🌐")
    st.markdown("Translate text between languages using a transformer model")
    
    # Sidebar for model settings
    st.sidebar.header("Model Settings")
    
    # Device selection
    device_option = st.sidebar.radio(
        "Select Device",
        ["CPU", "GPU (CUDA)"],
        index=1 if torch.cuda.is_available() else 0
    )
    device = "cuda" if device_option == "GPU (CUDA)" and torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        gpu_info = f"GPU: {torch.cuda.get_device_name(0)}"
        st.sidebar.info(f"Using {gpu_info}")
    else:
        st.sidebar.info("Using CPU")
    
    # Model path input
    model_path = st.sidebar.text_input("Model Path", DEFAULT_MODEL_PATH)
    src_tokenizer_path = st.sidebar.text_input("Source Tokenizer Path", DEFAULT_SRC_TOKENIZER_PATH)
    tgt_tokenizer_path = st.sidebar.text_input("Target Tokenizer Path", DEFAULT_TGT_TOKENIZER_PATH)
    
    # Initialize translator
    translator = get_translator(model_path, src_tokenizer_path, tgt_tokenizer_path, device)
    
    if translator is None:
        st.warning("Please check model paths and try again.")
        return
    
    # Main app content
    st.markdown("## Translation Interface")
    
    # Input tabs for different input methods
    input_method = st.radio("Select Input Method", ["Text Input", "File Upload", "Batch Translation"])
    
    # Translation parameters
    col1, col2 = st.columns(2)
    with col1:
        beam_size = st.slider("Beam Size", min_value=1, max_value=10, value=5, 
                             help="Higher values give better translations but are slower")
    
    with col2:
        show_attention = st.checkbox("Show Attention Visualization", value=False,
                                    help="Visualize attention weights (slows down translation)")
    
    # Process based on input method
    if input_method == "Text Input":
        source_text = st.text_area("Enter text to translate", height=150)
        
        if st.button("Translate"):
            if source_text:
                with st.spinner("Translating..."):
                    start_time = time.time()
                    
                    # Get attention weights if requested
                    if show_attention:
                        # First get the translation
                        translation = translator.translate(source_text, beam_size=beam_size)
                        
                        # Then get attention weights
                        attention_weights = translator.get_attention_weights(source_text)
                        
                        # Display translation
                        st.markdown("### Translation")
                        st.markdown(f"**Input:** {source_text}")
                        st.markdown(f"**Output:** {translation}")
                        
                        # Display attention visualization
                        st.markdown("### Attention Visualization")
                        
                        # Add controls for attention visualization
                        col1, col2 = st.columns(2)
                        with col1:
                            layer = st.slider("Layer", min_value=1, 
                                             max_value=len(attention_weights['encoder_decoder_attention']), 
                                             value=1)
                        with col2:
                            head = st.slider("Attention Head", min_value=1, 
                                            max_value=ModelConfig.num_heads, value=1)
                        
                        # TODO: Get source and target tokens
                        src_tokens = ["[Start]"] + source_text.split() + ["[End]"]
                        tgt_tokens = ["[Start]"] + translation.split() + ["[End]"]
                        
                        fig = plot_attention_weights(
                            attention_weights, 
                            src_tokens, 
                            tgt_tokens,
                            layer_idx=layer-1, 
                            head_idx=head-1
                        )
                        
                        if fig:
                            st.pyplot(fig)
                    else:
                        # Simple translation without attention
                        translation = translator.translate(source_text, beam_size=beam_size)
                        
                        # Display translation
                        st.markdown("### Translation")
                        st.markdown(f"**Input:** {source_text}")
                        st.markdown(f"**Output:** {translation}")
                    
                    end_time = time.time()
                    st.info(f"Translation completed in {end_time - start_time:.2f} seconds")
            else:
                st.warning("Please enter some text to translate")
    
    elif input_method == "File Upload":
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
        
        if uploaded_file is not None:
            content = uploaded_file.read().decode("utf-8")
            
            # Show file content
            st.markdown("### File Content Preview")
            st.text(content[:500] + ("..." if len(content) > 500 else ""))
            
            if st.button("Translate File"):
                with st.spinner("Translating file..."):
                    start_time = time.time()
                    
                    # Split content into lines
                    lines = [line.strip() for line in content.split("\n") if line.strip()]
                    
                    # Translate each line
                    translations = translator.translate_batch(lines, beam_size=beam_size)
                    
                    # Display results in a table
                    results = pd.DataFrame({
                        "Source": lines,
                        "Translation": translations
                    })
                    
                    st.markdown("### Translation Results")
                    st.dataframe(results)
                    
                    # Offer download option
                    st.download_button(
                        label="Download translations",
                        data=results.to_csv(index=False).encode('utf-8'),
                        file_name='translations.csv',
                        mime='text/csv',
                    )
                    
                    end_time = time.time()
                    st.info(f"Translated {len(lines)} lines in {end_time - start_time:.2f} seconds")
    
    elif input_method == "Batch Translation":
        # Input for batch translation
        batch_text = st.text_area("Enter multiple sentences (one per line)", height=200)
        
        if st.button("Translate Batch"):
            if batch_text:
                with st.spinner("Translating batch..."):
                    start_time = time.time()
                    
                    # Split into sentences
                    sentences = [line.strip() for line in batch_text.split("\n") if line.strip()]
                    
                    # Translate batch
                    translations = translator.translate_batch(sentences, beam_size=beam_size)
                    
                    # Display results
                    results = pd.DataFrame({
                        "Source": sentences,
                        "Translation": translations
                    })
                    
                    st.markdown("### Batch Translation Results")
                    st.dataframe(results)
                    
                    # Offer download option
                    st.download_button(
                        label="Download translations",
                        data=results.to_csv(index=False).encode('utf-8'),
                        file_name='batch_translations.csv',
                        mime='text/csv',
                    )
                    
                    end_time = time.time()
                    st.info(f"Translated {len(sentences)} sentences in {end_time - start_time:.2f} seconds")
            else:
                st.warning("Please enter some sentences to translate")
    
    # Add information about the project
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This is a neural machine translation system built with a Transformer architecture.
    
    The model was trained on parallel corpus data and uses beam search for better translation quality.
    
    [GitHub Repository](https://github.com/jmhasan1/Language_Translator_Transformer_Model)
    """)

if __name__ == "__main__":
    main()
