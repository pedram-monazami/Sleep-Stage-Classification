import torch.nn as nn
from .CNN_LSTM_Transformer import SingleCNNTransformer, DualCNNTransformer, SingleCNNLSTM, DualCNNLSTM
from .VisionTransformer import SingleViT, DualViT, DualCNNViT, DualResidualCNNViT


def get_model(config):
    """
    The main factory function for creating a model.
    """
    model_config = config['model']
    model_name = model_config['name']
    model_params = model_config.get('params', {})

    # --- Model Factory ---
    models = {
        'SingleViT': SingleViT,
        'DualViT': DualViT,
        'DualCNNViT': DualCNNViT,
        'DualResidualCNNViT': DualResidualCNNViT,
        'SingleCNNTransformer': SingleCNNTransformer,
        'DualCNNTransformer': DualCNNTransformer,
        'SingleCNNLSTM': SingleCNNLSTM,
        'DualCNNLSTM': DualCNNLSTM,
    }

    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found in model factory.")

    return models[model_name](**model_params)
