import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from lib.estimators import get_analytic_elbo, get_iwae_elbo
from lib.utils import get_cond_inds

class VariationalGRU(nn.Module):
    def __init__(self, input_dim, n_hidden, n_layer, dropout_p, cond_inds, final_activation, concat_cond_ts):
        super().__init__()
        self.cond_inds = cond_inds
        self.final_activation = final_activation
        self.n_hidden = n_hidden
        self.concat_cond_ts = concat_cond_ts

        if self.final_activation == 'softplus':
            final_a = nn.Softplus()
        elif self.final_activation == 'relu':
            final_a = nn.ReLU()
        elif self.final_activation == 'none':
            final_a = nn.Identity()

        n_inter_decode = len(cond_inds)

        self.encoder_gru = nn.GRU(input_size = input_dim, hidden_size = n_hidden*2, num_layers = n_layer,
                            dropout = dropout_p, batch_first = True, bidirectional = False)
        self.decoder_gru = nn.GRU(input_size = concat_cond_ts * n_inter_decode + 2, hidden_size = n_hidden + int(not concat_cond_ts) * n_inter_decode, 
                            num_layers = n_layer,
                            dropout = dropout_p, batch_first = True, bidirectional = False)
        self.output_fn = nn.Sequential(
                            nn.Linear(n_hidden, n_hidden//2),
                            nn.ReLU(),
                            nn.Linear(n_hidden//2, 2),
                            final_a
                            )

    def compute_loss_vae(self, x, pred_x, q_mean, q_logvar, noise_std=0.5, kl_weight = 1.):
        return get_analytic_elbo(x, pred_x, q_mean, q_logvar, noise_std, kl_weight)

    def compute_loss_iwae(self, x, pred_x, z0, q_mean, q_logvar, noise_std=0.5):
        return get_iwae_elbo(x, pred_x, z0, q_mean, q_logvar, noise_std)

    def forward(self, x, features, nsteps_encode, nsteps_decode, mask, n_elbo_samp, noise_std = 0.5, elbo_type="vae", *args, **kwargs):
        batch_size = x.shape[0]
        enc_input = torch.cat((x, features[:, :nsteps_encode, :]), dim = -1)
        if mask.ndim == 3:
            mask = mask[:,:, 0] # mask should be 2D
        seq_lengths = mask.sum(axis = 1)
        enc_input_packed = pack_padded_sequence(enc_input, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        enc_output_packed, hidden = self.encoder_gru(enc_input_packed)
        enc_output_unpacked, _ = pad_packed_sequence(enc_output_packed, batch_first=True, total_length = mask.shape[1])

        enc_pred = self.output_fn(enc_output_unpacked[:, :, :self.n_hidden])

        # hidden: (n_layers, batch_size, self.n_hidden*2)
        z_mean = hidden[:, :, :self.n_hidden ]
        z_logvar = hidden[:, :, self.n_hidden :]
        
        zs = []
        dec_preds = []
        for j in range(n_elbo_samp):
            epsilon = torch.randn(z_mean.size(), device=z_mean.device) 
            z = epsilon * torch.exp(0.5* z_logvar) + z_mean
            zs.append(z.reshape(batch_size, -1))
        
            if self.concat_cond_ts:
                decoder_input_i = torch.cat((enc_pred[:, -1, :], features[:, nsteps_encode, self.cond_inds]), 1).unsqueeze(1)
                z_i = z
            else:
                decoder_input_i = enc_pred[:, -1, :].unsqueeze(1)
                z_i = torch.cat((z, features[:, nsteps_encode, self.cond_inds].unsqueeze(0).repeat(z.shape[0], 1, 1)), 2)

            dec_preds_j = [enc_pred[:, -1, :].unsqueeze(1)]
            for di in range(nsteps_decode - 1):
                decoder_output, decoder_hidden = self.decoder_gru(decoder_input_i, z_i)
                if not self.concat_cond_ts:
                    decoder_output = decoder_output[:, :, :self.n_hidden]
                decoder_output = self.output_fn(decoder_output)
                dec_preds_j.append(decoder_output)

                z_i = decoder_hidden
                assert(decoder_output.shape[1] == 1)
                if self.concat_cond_ts:
                    decoder_input_i = torch.cat((decoder_output[:, 0, :], features[:, nsteps_encode+di, self.cond_inds]), 1).unsqueeze(1)
                else:
                    decoder_input_i = decoder_output
            pred_x = torch.cat(dec_preds_j, dim = 1)
            dec_preds.append(pred_x)

        if elbo_type == 'vae':
            loss_func = lambda t, y: self.compute_loss_vae(t, y, z_mean.reshape(batch_size, -1), z_logvar.reshape(batch_size, -1), noise_std)
        elif elbo_type == 'iwae':
            loss_func = lambda t, y: self.compute_loss_iwae(t, y, torch.stack(zs).transpose(0, 1), z_mean.reshape(batch_size, -1), z_logvar.reshape(batch_size, -1), noise_std)
        
        return (torch.stack([enc_pred[:, :-1, :]] * n_elbo_samp), torch.stack(dec_preds),
                    loss_func)

def load_model_vgru(all_features, config_dict):
    return VariationalGRU(input_dim = len(all_features) +2,
                            n_hidden = config_dict['n_units'],
                            n_layer = config_dict['n_layer'],
                            dropout_p = config_dict['dropout_p'],
                            cond_inds = get_cond_inds(all_features, config_dict), 
                            final_activation = config_dict['final_activation'],
                            concat_cond_ts=config_dict['concat_cond_ts'])
