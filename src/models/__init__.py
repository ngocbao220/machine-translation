# src/models/__init__.py
from .base_transformer import TransformerTranslation
# from .re_transformer import ReTransformerTranslation

def build_model(config, vocab_size, pad_idx):
    model_type = config['model_type'] # 'base' hoặc 're'
    
    # Các tham số chung
    kwargs = {
        'src_vocab_size': vocab_size,
        'tgt_vocab_size': vocab_size,
        'd_model': config['d_model'],
        'nhead': config['nhead'],
        'num_encoder_layers': config['num_layers'],
        'num_decoder_layers': config['num_layers'],
        'dim_feedforward': config['dim_feedforward'],
        'dropout': config['dropout'],
        'pad_idx': pad_idx
    }

    if model_type == 'base':
        return TransformerTranslation(**kwargs)
    elif model_type == 're':
        # Nếu ReTransformer có thêm tham số riêng, thêm vào đây
        # kwargs['extra_param'] = config['extra']
        # return ReTransformerTranslation(**kwargs)
        pass
    else:
        raise ValueError(f"Unknown model type: {model_type}")