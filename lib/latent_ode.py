import torch
import torch.nn as nn

from lib.estimators import get_analytic_elbo, get_iwae_elbo
from lib.latent_ode_helper import ODEFuncNN, GRU, EncoderGRUODE, NeuralODE, LatentNeuralODE
from lib.utils import get_cond_inds


class AutoregressiveDecoder(nn.Module):
    def __init__(self, input_dim, state_dim=2, final_activation='none'):
        super().__init__()

        self.final = nn.Linear(input_dim + state_dim, state_dim, bias=False)
        with torch.no_grad(): # helpful initialization
            self.final.weight[0, 0] = 1.01
            self.final.weight[1, 1] = 1.0
        self.final_activation = final_activation

    def forward(self, init_states, z, all_states=None, features=None):
        """
        init_states: (batch_size, state_dim)
        z: (batch_size, n_times, input_dim)
        all_states: (batch_size, n_times, state_dim). Actual state values at all times.
        features: (batch_size, n_times, feat_dim). Intervention features.
        """
        cur_states = init_states
        preds = [init_states]

        for i in range(z.shape[1] - 1):
            cur_input = torch.cat([cur_states, z[:, i, :]], dim=-1)

            if features is not None:
                cur_input = torch.cat([cur_input, features[:, i, :]], dim=-1)

            cur_states = self.final(cur_input)

            if self.final_activation == 'softplus':
                cur_states = nn.Softplus()(cur_states)
            elif self.final_activation == 'relu':
                cur_states = nn.ReLU()(cur_states)
            else:
                pass

            preds.append(cur_states)

            # replace predicted with actual state for next time step
            if all_states is not None and i != z.shape[1] - 1:
                cur_states = all_states[:, i+1, :]

        return torch.stack(preds, dim=1)


class EncoderGRUODEForward(EncoderGRUODE):
    """GRU with hidden dynamics represented by Neural ODE.
    Implements the GRU-ODE model in: https://arxiv.org/abs/1907.03907.
    Observations are encoded by a RNN/GRU. Between observations, the hidden
    state is evolved using a Neural ODE.

    Attributes:
        gru (nn.Module): GRU unit used to encode input data.
        node (nn.Module): Neural ODE used to evolve hidden dynamics.
        out (nn.Module): NN mapping from hidden state to output.
        latent_dim (int): Dimension of latent state.
    """

    def __init__(self, latent_dim, rec_gru, rec_node, rec_output):
        """Initialize GRU-ODE model.
        This module is intended for use as the encoder of a latent NODE.
        Args:
            latent_dim (int): Dimension of latent state.
            rec_gru (nn.Module): GRU used to encoder input data.
            rec_node (nn.Module): NODE used to evolve state between GRU units.
            rec_output (nn.Module): Final linear layer.
        """
        super().__init__(latent_dim, rec_gru, rec_node, rec_output)

    def get_last_tp(self, mask):
        return [-1] * mask.shape[0]


class LatentNeuralODEForward(LatentNeuralODE):
    '''
    A latent ODE that runs the encoder forward nsteps_encode timesteps to get the latent state
        Then decodes
    '''
    def __init__(self, enc, nodef, dec, all_features, cond_inds, concat_cond_ts, use_actual_states_in_decoder = True):
        """Initialize latent neural ODE.
        Args:
            dec (nn.Module): Decoder module.
            enc (nn.Module): Encoder module.
            nodef (nn.Module): Neural ODE module.
            cond_inds (torch.Tensor): Indices of the features used as conditional latent variables.
            concat_cond_ts (bool): Whether to concatenate interventions to z0 or to all z in decode
        """
        super().__init__(enc, nodef, dec)
        self.all_features = all_features # list of strings
        self.use_actual_states_in_decoder = use_actual_states_in_decoder
        self.cond_inds = cond_inds
        self.concat_cond_ts = concat_cond_ts

    def get_latent_initial_state(self, x, ts, mask=None):
        """Compute latent parameters.
        Allows masking via a 2D binary array of shape (B x T).
        Args:
            x (torch.Tensor): Data points.
            ts (torch.Tensor): Timepoints of observations.
            mask (torch.Tensor, optional): Masking array.
        Returns:
            torch.Tensor, torch.Tensor: Latent mean and logvar parameters.
        """
        out = self.enc.forward(x, ts, mask)

        qz0_mean = out[:, :out.size(1) // 2]
        qz0_logvar = out[:, out.size(1) // 2:]

        return qz0_mean, qz0_logvar

    def forward(self, encode_x, features, nsteps_encode, nsteps_decode, mask=None,
                rtol=1e-3, atol=1e-4, noise_std=0.5, n_elbo_samp=1, elbo_type="vae",
                stl=False):
        """Compute forward pass of Latent NODE.

        TODO: The number of estimator arguments is growing. I think the best way to refactor
        is to let them be handled at the training level, right now noise_std is flowing
        through this method anyways.

        Args:
            encode_x (torch.Tensor): Input states
            features (torch.Tensor): feature matrix; fed into encoder
            nsteps_encode (int): Number of time steps in the encode portion.
            nsteps_decode (int): Number of time steps in decode portion.
            mask (torch.Tensor, optional): 1 where x is observed, 0 where it is not; should be batch_size x n_times
            rtol (float, optional): NODE ODE solver relative tolerance.
            atol (float, optional): NODE ODE solver absolute tolerance.
            noise_std (float, optional): Standard deviation used to calculate gaussian pdf.
            n_elbo_samp (int): Number of samples taken to approximate ELBO.
            elbo_type (str): Type of ELBO used to evaluate log px.
            stl (boolean): Uses Sticking-The-Landing estimator.
        """
        assert(encode_x.shape[1] == nsteps_encode)
        assert(features.shape[1] == nsteps_encode + nsteps_decode)

        encode_features = features[:, :nsteps_encode, :]
        pred_features = features[:, nsteps_encode:nsteps_encode+nsteps_decode, :]

        encoder_input = torch.cat((encode_x, encode_features), dim=-1)

        ts_encode = torch.arange(nsteps_encode, device=encode_x.device).float()
        ts_decode = torch.arange(nsteps_decode, device=encode_x.device).float()

        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, :nsteps_encode]
            elif mask.ndim == 3:
                mask = mask[:, :nsteps_encode, 0]

        qz0_mean, qz0_logvar = self.get_latent_initial_state(encoder_input, ts_encode, mask)

        batch_size = qz0_mean.shape[0]
        qz0_mean = torch.repeat_interleave(qz0_mean, n_elbo_samp, 0)
        qz0_logvar = torch.repeat_interleave(qz0_logvar, n_elbo_samp, 0)
        encode_x = torch.repeat_interleave(encode_x, n_elbo_samp, 0)

        z0, epsilon = self.reparameterize(qz0_mean, qz0_logvar)

        z0_aug = z0.clone()

        if self.cond_inds is not None and len(self.cond_inds):
            # TODO: We could try using the mean of future features.
            cond_vars = features[:, nsteps_encode, self.cond_inds]
            cond_vars = torch.repeat_interleave(cond_vars, n_elbo_samp, 0)
            z0_aug = torch.cat([z0, cond_vars], -1)

        enc_cond = pred_cond = None

        if self.concat_cond_ts:
            enc_cond = encode_features[:, :, self.cond_inds]
            pred_cond = pred_features[:, :, self.cond_inds]

            enc_cond = torch.repeat_interleave(enc_cond, n_elbo_samp, 0)
            pred_cond = torch.repeat_interleave(pred_cond, n_elbo_samp, 0)

        enc_states = encode_x if self.use_actual_states_in_decoder else None

        ts_reconstr = torch.flip(torch.arange(nsteps_encode + 1, device=encode_x.device).float(), [0])

        reconstr_z = self.generate_from_latent(z0_aug, ts_reconstr, rtol, atol)
        reconstr_z = torch.flip(reconstr_z[:, 1:, :], [1])

        reconstr_x = self.dec(encode_x[:, 0, :], reconstr_z, enc_states, enc_cond)

        pred_z = self.generate_from_latent(z0_aug, ts_decode, rtol, atol)
        pred_x = self.dec(reconstr_x[:, -1, :], pred_z, features=pred_cond)

        # Duplicated in pred_x.
        reconstr_x = reconstr_x[:, :-1, :]

        reconstr_x = reconstr_x.view(batch_size, n_elbo_samp, *reconstr_x.shape[1:])
        pred_x = pred_x.view(batch_size, n_elbo_samp, *pred_x.shape[1:])

        qz0_mean = qz0_mean.view(batch_size, n_elbo_samp, *qz0_mean.shape[1:])
        qz0_logvar = qz0_logvar.view(batch_size, n_elbo_samp, *qz0_logvar.shape[1:])

        if elbo_type == "vae":
            qz0_mean, qz0_logvar = qz0_mean.transpose(0, 1), qz0_logvar.transpose(0, 1)
            elbo_func = lambda t, y: get_analytic_elbo(t, y, qz0_mean, qz0_logvar, noise_std, stl)
        elif elbo_type == "iwae":
            z0 = z0.view(batch_size, n_elbo_samp, *z0.shape[1:])
            elbo_func = lambda t, y: get_iwae_elbo(t, y, z0, qz0_mean, qz0_logvar, noise_std, stl)
        else:
            raise NotImplementedError

        return reconstr_x.transpose(0, 1), pred_x.transpose(0, 1), elbo_func


def load_model_lode(all_features, config_dict):
    cond_inds = get_cond_inds(all_features, config_dict)   
    cond_dim = len(cond_inds)
    concat_cond_ts = config_dict['concat_cond_ts']

    obs_dim = 2 + len(all_features)
    rec_latent_dim = config_dict['latent_dim'] * 2
    node_latent_dim = config_dict['latent_dim']
    rec_gru_unit = config_dict['n_units']
    rec_node_hidden = config_dict['n_units']
    rec_node_layer = config_dict['n_layer']
    rec_node_act = 'Tanh'
    rec_out_hidden = config_dict['n_units']
    latent_node_hidden = config_dict['n_units']
    latent_node_layer = config_dict['n_layer']
    latent_node_act = 'Tanh'

    dec_type = config_dict['dec_type']
    output_dim = 2

    enc_gru = GRU(rec_latent_dim, obs_dim, rec_gru_unit)

    enc_node = ODEFuncNN(rec_latent_dim * 2, rec_node_hidden,
                         rec_node_layer, rec_node_act)

    enc_out = nn.Sequential(
        nn.Linear(rec_latent_dim * 2, rec_out_hidden),
        nn.Tanh(),
        nn.Linear(rec_out_hidden, node_latent_dim * 2)
    )

    enc = EncoderGRUODEForward(rec_latent_dim, enc_gru, enc_node, enc_out)

    latent_node = NeuralODE(node_latent_dim + cond_dim,
                            latent_node_hidden, latent_node_layer,
                            latent_node_act) # TODO: check whether dims are correct

    decoder_input_dim = node_latent_dim + cond_dim + cond_dim * int(concat_cond_ts)

    if dec_type == "AR":
        dec = AutoregressiveDecoder(decoder_input_dim, final_activation=config_dict['final_activation'])
    elif dec_type == "NN":
        raise NotImplementedError('Not currently supported.')
        # TODO: If we want this, we need to change the forward function.
        # dec = nn.Sequential(nn.Linear(decoder_input_dim, 100), nn.ReLU(), nn.Linear(100, output_dim))
    else:
        raise NotImplementedError('Unknown Decoder.')

    return LatentNeuralODEForward(enc, latent_node, dec, all_features, cond_inds, concat_cond_ts)
