import torch
import torch.nn as nn
import math

# Input Embedding

class InputEmbeddings(nn.Module):

    # d_model - dimension of the model (vector ) as per transformer
    #vocab_size = no of words in the vocabulary

    def __init__(self, d_model : int , vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # In Pytorch 
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Embedding maps a number(input id) to a vector of model dimension(here 512)

    def foward(self , x):
        return self.embedding(x)*math.sqrt(self.d_model)
    #  In the embedding layers, we multiply those weights by √dmodel.

# Positional Embedding

class PositionalEmbeddings(nn.Module):

    # seq_len = maximum length of the sentences
    #dropout = to make model less overfitting

    def __init__(self, d_model: int, seq_len: int , dropout: float)->None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Because positional encodings is a vector of size d_model and leght of each input sequence is seq_len
        # Create a matrix of shape (seq_len , d_model)
        pe = torch.zeros(seq_len , d_model)

        # Create a vector of shape (seq_len , 1)
        position = torch.arange(0 , seq_len , dtype=torch.float).unsqueeze(1)

        # we calculate in log space for numerical stability
        div_term = torch.exp(torch.arange(0 , d_model,2).float() * (-math.log(10000.0)/d_model))

        # Apply the sin / cos to even/odd positions

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)   # (1 , seq_len , d_model)

        self.register_buffer('pe', pe)
        # register_buffer is a way to keep a tensor that won't be updated during the training process.
        
    # In the positional embedding layers, we add the positional encoding to the input embeddings.
    def forward(self,x):
        x = x + (self.pe[:, :x.shape[1] , :]).requires_grad_(False)
        return self.dropout(x)
        # require_grad_(False) will make sure that the perticular tensor is not updated / learned during training
    

# We add epsilon to add numerical stability and also avoid division by zero

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6)->None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))    #Multiplicative parameter
        self.bias = nn.Parameter(torch.zeros(1))    # Additive parameter
        # nn.Parameter makes parameter learnable
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # mean parameter of last dimension 
        std = x.std(dim=-1, keepdim=True)    # std deviation of last dimension
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

# Feed Forward is a fully connected network that the model uses both in encoder and decoder

class FeedForwardBlock(nn.Module):

    def __init__(self , d_model : int, d_ff: int, dropout:float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) #  W1 and B1 , bias is by default added
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff , d_model) # W2 and B2 , bias is by default added

    def forward(self, x):
        # (Batch , seq_len , d_model) ---> (Batch , seq_len , d_ff) ---> (Batch , seq_len , d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self , d_model:int, h : int, dropout : float) -> None :
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0 , "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model , d_model)     # Wq
        self.w_k = nn.Linear(d_model, d_model)      # Wk
        self.w_v = nn.Linear(d_model , d_model)     # Wv

        self.w_o = nn.Linear(d_model, d_model)      # Wo
        self.dropout = nn.Dropout(dropout)

    
    def attention(query, key , value ,mask ,dropout = nn.Dropout):
        d_k = query.shape[-1]

        # (Batch  , h , seq_len , d_k) ---> (Batch , h , seq_len , seq_len)
        attention_scores = (query @ key.transpose(-2 , -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0 , -1e9)
        attention_scores = attention_scores.softmax(dim = -1)   # (Batch , h , seq_len , seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value) , attention_scores


    
    def forward(self, q , k , v , mask):
        query =self.w_q(q)     #(Batch , seq_len , d_model) ---> (Batch , seq_len , d_model)
        key = self.w_k(k)      #(Batch , seq_len , d_model) ---> (Batch , seq_len , d_model)
        value = self.w_v(v)    #(Batch , seq_len , d_model) ---> (Batch , seq_len , d_model)
        
        # Reshape the tensor matrices to perform the calculations
        # (Batch , seq_len , d_model) ---> (Batch , seq_len , h , d_k) ---> (Batch , h , seq_len , d_k)
        query = query.view(query.shape[0],query.shape[1], self.h , self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1], self.h , self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1], self.h , self.d_k).transpose(1,2)

        # Calculate attention
        x , self.attention_scores = MultiHeadAttentionBlock.attention(query, key , value, mask , self.dropout)

        # Combine all the heads together
        # (Batch , h , seq_len , d_k) ---> (Batch , seq_len , h , d_k)--> (Batch , seq_le, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0] , -1 , self.h * self.d_k)

        # (Batch , seq_len , d_model) ---> (Batch , seq_le, d_model)
        return self.w_o(x)
    
    # ** here we have not perfomred cocatenation of heads so far above 


class ResidualConnection(nn.Module):

    def __init__(self , dropout : float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self , x , sublayer):
        # return self.norm(x + self.dropout(sublayer(x)))
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):

    def __init__(self , self_attention_block : MultiHeadAttentionBlock , feed_forward_block : FeedForwardBlock , dropout : float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        # Residual Connections
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers : nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        super.norm = LayerNormalization()
    
    def forward(self, x, mask):
        # # x.shape = (Batch_size, seq_len, d_model)
        # x = self.norm(x)
        # for layer in self.layers:
        #     x = layer(x, mask)
        # return x

        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)  # (Batch_size, seq_len, d_model)
    

class DecoderBlock(nn.Module):

    def __init__(self , self_attention_block : MultiHeadAttentionBlock , cross_attention_block : MultiHeadAttentionBlock , feed_forward_block : FeedForwardBlock , dropout : float) ->None :
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self , x , encoder_output , src_mask , target_mask ):
        x = self.residual_connections[0](x , lambda x : self.self_attention_block(x , x , x , target_mask))
        x = self.residual_connections[1](x , lambda x : self.cross_attention_block(x , encoder_output , encoder_output , src_mask))
        x = self.residual_connections[2](x , self.feed_forward_block)
        return x 
class Decoder(nn.Module):

    def __init__(self, layers : nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        super.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm(x)  # (Batch_size, seq_len, d_model)

class PrjectionLayer(nn.Module):

    def __init__(self , d_model : int , vocab_size : int)->None:
        super().__init__()
        self.linear_proj = nn.Linear(d_model, vocab_size)

    def forward(self , x):
        return torch.log_softmax(self.linear_proj(x) , dim=-1)


class Transformer(nn.Module):

    def __init__(self , encoder : Encoder , decoder : Decoder , src_embed : InputEmbeddings , target_embed : InputEmbeddings , src_pos : PositionalEmbeddings , target_pos : PositionalEmbeddings , projection_layer :PrjectionLayer) -> None:
        super.__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer

    def encode(self , src , src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self , encoder_output , src_mask , target , target_mask):
        target = self.target_embed(target)
        target = self.target_pos(target)
        return self.decoder(target, encoder_output, src_mask, target_mask)
    
    def project(self , x):
        return self.projection_layer(x)

# Method to combine all above blocks together
# Given all the hyperparametrs of transformers ,  will build transfomer for us initializing all encoders , decoders , embedding etc.
# d_ff = hidden layers of feed forward network layer
def build_transformer( src_vocab_size : int , tgt_vocab_size : int , src_seq_len : int , tgt_seq_len : int , d_model : int = 512 , N : int = 6 , h : int = 8 , dropout : float = 0.1 , d_ff = 2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model , src_vocab_size)
    tgt_embed = InputEmbeddings(d_model , tgt_vocab_size)

    #  Create the positional embedding layer
    src_pos = PositionalEmbeddings(d_model , src_seq_len , dropout)
    tgt_pos = PositionalEmbeddings(d_model , tgt_seq_len , dropout)

    # Create the encoder blocks 
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model , h , dropout)
        feed_forward_block = FeedForwardBlock(d_model , d_ff , dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block , feed_forward_block ,dropout)
        encoder_blocks.append(encoder_block)
    
    # Create the decoder blocks\
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model , h , dropout)
        cross_attention_block = MultiHeadAttentionBlock(d_model , h , dropout)
        feed_forward_block = FeedForwardBlock(d_model , d_ff , dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block , cross_attention_block , feed_forward_block , dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(encoder_blocks)
    decoder = Decoder(decoder_blocks)

    # Create the projection layer
    projection_layer = PrjectionLayer(d_model , tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder , decoder , src_embed , tgt_embed , src_pos , tgt_pos , projection_layer)
    
    # Initialize the parameters to make training faster
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer



