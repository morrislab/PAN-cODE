import torch
import torch.nn as nn
from lib.latent_ode import load_model_lode
from lib.gru_ode import load_model_gruode
from lib.gru import load_model_gru
from lib.variational_gru import load_model_vgru

'''
Writing a new model to be compatible with train.py

1. forward function
    Inputs:
        - time series matrix for encode portion; (batch_size, t_encode, 2)
        - features for encode and decode portion; (batch_size, t_encode + t_decode, n_features)
        - t_encode; int
        - t_decode; int
        - mask: 1 where data is observed; 0 otherwise; (batch_size, t_encode)
        - n_elbo_samp; int

    Outputs:
        - reconstr_x: reconstruction for encode portion; (n_elbo_samp, batch_size, t_encode-1, 2)
            - If the input sequence is given by [x_0, x_1, ..., x_{t_encode-1}], then this function should output
                [x'_1, x'_2, ..., x'_{t_encode-1}]
        - pred_x: predicted forecast for decode portion; (n_elbo_samp, batch_size, t_decode, 2)
            - This should output [x'_{t_encode}, ..., x'_{t_encode + t_decode}]
        - loss_func: a function that takes in
                - actual values; (batch_size, t_encode, 2)
                - predicted values; (batch_size, t_encode, 2)
            and returns
                - a scalar loss
            
2. Write a load_model function that takes in all_feature and config_dict and returns the model object

'''

def criterion_func(mask, y_pred_known, b_ts_mat):
    criterion = nn.MSELoss(reduction = 'sum')
    masked_pred, masked_ts = (y_pred_known * mask.unsqueeze(-1), b_ts_mat * mask.unsqueeze(-1))
    return criterion(masked_pred, masked_ts)

def load_model(all_features, config_dict):
    if config_dict['model'] == 'lode':
        return load_model_lode(all_features, config_dict)
    elif config_dict['model'] == 'gruode':
        return load_model_gruode(all_features, config_dict)
    elif config_dict['model'] == 'gru':
        return load_model_gru(all_features, config_dict)
    elif config_dict['model'] == 'vgru':
        return load_model_vgru(all_features, config_dict)
    else:
        raise NotImplementedError
