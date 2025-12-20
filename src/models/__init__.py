# src/models/__init__.py
from .base_transformer import TransformerTranslation
from .re_transformer import ReTransformerTranslation

def build_model(config, vocab_size, pad_idx):
    model_type = config['model_type'] # 'base' hoặc 're'
    
    # Các tham số chung từ config.yaml
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
        # ReTransformer sử dụng cùng bộ tham số nhưng kiến trúc encoder khác
        return ReTransformerTranslation(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")