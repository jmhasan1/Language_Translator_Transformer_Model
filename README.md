<h2>Language Translator Using Transformer model </h2>
<P>This is a comprehensive implementation of the Transformer architecture as described in the "Attention Is All You Need" paper. Breaking down the model : <br>

<H4>Input Embeddings</H4>:
The InputEmbeddings class converts input tokens into dense vectors. It uses PyTorch's nn.Embedding layer to map each token to a vector of size d_model. The vectors are then scaled by √(d_model) as per the original paper.<br>
<h4>Positional Embeddings:</h4>
The PositionalEmbeddings class adds position information to the input embeddings. It creates a matrix of positional encodings using sine and cosine functions. These encodings are added to the input embeddings to give the model information about the sequence order.
Layer Normalization:<br>
The LayerNormalization class implements layer normalization, which helps stabilize the learning process. It normalizes the inputs across the features.
Feed Forward Block:
The FeedForwardBlock class implements the position-wise feed-forward networks used in both the encoder and decoder. It consists of two linear transformations with a ReLU activation in between.
Multi-Head Attention Block:
The MultiHeadAttentionBlock class implements the multi-head attention mechanism. It projects the input into query, key, and value vectors, splits them into multiple heads, computes scaled dot-product attention, and concatenates the results.
Residual Connection:
The ResidualConnection class implements the residual connections used throughout the model. It applies layer normalization, then the sublayer (attention or feed-forward), adds the result to the input, and applies dropout.
Encoder Block:
The EncoderBlock class combines self-attention and feed-forward layers with residual connections to form a single encoder layer.
Encoder:
The Encoder class stacks multiple encoder blocks and applies a final layer normalization.
Decoder Block:
The DecoderBlock class is similar to the encoder block but includes an additional cross-attention layer that attends to the encoder output.
Decoder:
The Decoder class stacks multiple decoder blocks and applies a final layer normalization.
Projection Layer:
The ProjectionLayer class converts the decoder output to logits over the target vocabulary.
Transformer:
The Transformer class combines all the above components into a complete model. It includes methods for encoding, decoding, and projecting.
build_transformer Function:
This function is a convenient way to construct the entire Transformer model given hyperparameters like vocabulary sizes, sequence lengths, model dimension, number of layers, etc.

The code follows the original Transformer architecture closely, including details like scaling the embeddings, using layer normalization, and initializing parameters with Xavier uniform initialization.</P>
