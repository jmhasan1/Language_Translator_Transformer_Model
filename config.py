def get_config():
    return {
        "batch_size": 4,  # Reduced from 8
        "num_epochs": 10,  # Reduced from 20
        "lr": 10**-4,  # Keep as is
        "seq_len": 256,  # Reduced from 350
        "d_model": 256,  # Reduced from 512
        "datasource": 'opus_books',
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }