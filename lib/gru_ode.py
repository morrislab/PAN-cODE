import torch
import torch.nn as nn

from torchdiffeq import odeint_adjoint as odeint
from lib.latent_ode_helper import GRU, ODEFuncNN


class EncoderGRUODE(nn.Module):
    def __init__(self, latent_dim, gru, nodef, out):
        super().__init__()

        self.gru = gru
        self.nodef = nodef
        self.out = out
        self.latent_dim = latent_dim

    def reconstruct(self, x, features, mask):
        enc_data = torch.cat([x, features], dim=-1)

        tps = torch.arange(x.size(1)).to(x.device)
        # Insert dummy time point which is discarded later.
        tps = torch.cat(((tps[0]-0.01).unsqueeze(0), tps), 0)

        h = torch.zeros(x.size(0), self.latent_dim * 2).to(x.device)

        if mask is not None:
            h_arr = torch.zeros(x.size(0), x.size(1), h.size(1)).to(x.device)
            r_fill_mask = self.right_fill_mask(mask)

        for i in range(x.size(1)):
            # Masks out ode evolutions
            if i != 0:
                h_ode = odeint(self.nodef, h, tps[i:i+2])[1]

                if mask is not None:
                    curr_rmask = r_fill_mask[:, i].view(-1, 1)
                    h_ode = h_ode * curr_rmask + h * (1 - curr_rmask)
            else:
                # Don't evolve hidden state prior to first observation
                h_ode = h

            # Masks out gru updates
            h_rnn = self.gru(enc_data[:, i, :], h_ode)

            if mask is not None:
                curr_mask = mask[:, i].view(-1, 1)
                h = h_rnn * curr_mask + h_ode * (1 - curr_mask)

                h_arr[:, i, :] = h
            else:
                h = h_rnn

        return self.out(h_arr), h_arr

    def predict(self, x_init, hid_init, features):
        tp = torch.Tensor([0., 1.]).to(x_init.device)

        h = hid_init
        curr_x = x_init

        out_arr = torch.zeros(x_init.size(0), features.size(1), x_init.size(1))
        out_arr = out_arr.to(x_init.device)

        for i in range(features.size(1)):
            h = odeint(self.nodef, h, tp)[1]

            curr_feat = features[:, i, :]
            curr_in = torch.cat([curr_x, curr_feat], -1)

            h = self.gru(curr_in, h)

            curr_in = self.out(h)
            out_arr[:, i, :] = curr_in

        return out_arr

    def forward(self, x, features, t_enc, t_dec, mask, n_elbo_samp, *args, **kwargs):
        enc_feat = features[:, :t_enc, :]

        enc_mask = None
        if mask is not None:
            if mask.ndim == 2:
                enc_mask = mask[:, :t_enc]
            elif mask.ndim == 3:
                enc_mask = mask[:, :t_enc, 0]

        enc_out, enc_hidden = self.reconstruct(x, enc_feat, enc_mask)

        last_hidden = enc_hidden[:, -1, :]
        last_out = enc_out[:, -1, :]

        pred_features = features[:, -t_dec:, :]
        pred_out = self.predict(last_out, last_hidden, pred_features)

        enc_out = enc_out[:, :-1, :].unsqueeze(0)
        pred_out = pred_out.unsqueeze(0)

        return torch.repeat_interleave(enc_out, n_elbo_samp, 0), torch.repeat_interleave(pred_out, n_elbo_samp, 0), lambda t, y: nn.MSELoss()(t, y)

    def right_fill_mask(self, mask):
        """Return mask with all non-leading zeros filled with ones."""
        mask = mask.detach().clone()
        for i, row in enumerate(mask):
            seen = False
            for j, mp in enumerate(row):
                if seen:
                    mask[i][j] = 1
                elif mp == 1:
                    mask[i][j] = 0
                    seen = True
        return mask


def load_model_gruode(all_features, config_dict):
    obs_dim = 2 + len(all_features)
    latent_dim = config_dict['latent_dim']

    gru_unit = config_dict['n_units']
    node_hidden = config_dict['n_units']
    node_layer = config_dict['n_layer']
    node_act = 'Tanh'

    out_hidden = config_dict['n_units']

    gru = GRU(latent_dim, obs_dim, gru_unit)

    nodef = ODEFuncNN(latent_dim * 2, node_hidden, node_layer, node_act)

    out = nn.Sequential(
        nn.Linear(latent_dim * 2, out_hidden),
        nn.Tanh(),
        nn.Linear(out_hidden, 2)
    )

    return EncoderGRUODE(latent_dim, gru, nodef, out)
