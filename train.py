import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader , random_split

from dataset import BilingualDataset , causal_mask
# from dataset import BilingualDataset 
from model import build_transformer

from config import get_weights_file_path , get_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

# from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

import warnings
from tqdm import tqdm
from pathlib import Path

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()


def get_all_sentences(ds , lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config , ds , lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))    # if our tokenizers sees a word that it doesn't recognize in it's vocabulary will replace it with 'UNK' , i.e map corrosponding number to the word with 'UNK'
        # Basically we split by white sapce
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UNK]" , "[PAD]" , "[SOS]", "[EOS]"] , min_frequrncy = 2)
        tokenizer.train_from_iterator(get_all_sentences(ds , lang) , trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer
                       
# to load the dataset
def get_ds(config):
    # parameters :  name of datset , subset (configurable ) , what split want to use
    ds_raw = load_dataset('opus_books' , f'{config["lang_src"]}-{config["lang_tgt"]}' , split='train')

    # Build tokenizers 
    tokenizer_src = get_or_build_tokenizer(config , ds_raw , config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config , ds_raw , config['lang_tgt'])

    # keeep 90% for training and 10 % for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw , val_ds_raw = random_split(ds_raw , [train_ds_size , val_ds_size])
    # random_split (Pytorch) allows to split dataset using the size given as input

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

     # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds , batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds , batch_size= 1 , shuffle=True)

    return train_dataloader , val_dataloader , tokenizer_src, tokenizer_tgt

# will build according our vocabulary size a transformer model 
def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

# Tensor allows to visualize the loss
def train_model(config):
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    Path(config['model_folder']).mkdir(parents=True , exist_ok=True)

    train_dataloader , val_daloader , tokenizer_src , tokenizer_tgt =  get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard 
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optimizer.Adam(model.parameters() , lr = config['lr'], eps =  1e-9 )

    # to resume the training in case of modle crash , restore the state of the model and the state of the optimizer

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config , config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_stae_dict(state['optimizer_stae_dict'])
        global_step = state['global_step']
    

    # ignore_index to ignore padding tokens as we dont want to add padding tokens to loss
    # lebel smootheing make model less confidence of it's decision , less overfit
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAID]') , label_smoothing=0.1).to(device)

    # traing loop
    for epoch in range(initial_epoch , config['num_epochs'] ):
        model.train()
        batch_iterator = tqdm(train_dataloader , desc=f'Processing epoch {epoch : 02d}')
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
        
        
        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)



if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)