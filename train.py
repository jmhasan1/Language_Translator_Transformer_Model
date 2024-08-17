import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader , random_split

from dataset import BilingualDataset , causal_mask

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_all_sentences(ds , lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config , ds , lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))    # if our tokenizers sees a word that it doesn't recognize in it's vocabulary will replace it with 'UNK' , i.e map corrosponding number to the word with 'UNK'
        # Basically we split by white sapce
        tokenizer.pre_tokenizer = Whitespace
        trainer = WordLevelTrainer(special_tokens = ["[UNK]" , "[PAD]" , "[SOS]", "[EOS]"] , min_frequrncy = 2)
        tokenizer.train_from_iterator(get_all_sentences(ds , lang) , trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer
                       
# to load the dataset
def get_ds(config):
    # parameters :  name of datset , subset (configurable ) , what split want to use
    ds_raw = load_dataset('opus_books' , f'{config["lang_src"]} - {config["lang_tgt"]}' , split='train')

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


