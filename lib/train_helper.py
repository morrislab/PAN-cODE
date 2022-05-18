from tqdm.auto import tqdm
import torch
import torch.nn as nn
import numpy as np

def isin(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)

def mse(A, B):
    return np.square(np.subtract(A, B)).mean()

def mae(A, B):
    return np.abs(np.subtract(A, B)).mean()

def train_epoch(train_val_loader, net, optimizer, device, nsteps_decode, n_elbo_samp, elbo_type,
                reconstr_weight, criterion_func, noise_std, train_on_val=False,
                clip_value=None, shift=0):
    all_actual_encode, all_actual_decode, all_reconstr, all_pred, all_country_ids = [], [], [], [], []
    total_train_pred_loss, total_train_reconstr_loss = 0, 0
    num_train_pred_elements, num_train_reconstr_elements = 0, 0
    total_train_loss = 0

    for batch in train_val_loader:
        # train
        net.train()
        optimizer.zero_grad()

        b_country_id, b_ts_mat, b_mask, b_features, b_nsteps_encode, b_nsteps_decode = batch

        b_ts_mat = b_ts_mat.float().to(device)
        b_features = b_features.float().to(device)
        b_mask = b_mask.to(device)

        bsize = len(b_country_id)

        if train_on_val:  # encode on the training part; decode on the validation part
            train_mask = isin(b_mask, torch.tensor([2, 1]).to(device)).int()
            nsteps_encode = b_nsteps_encode[0] + b_nsteps_decode[0]
            nsteps_decode = (b_mask == 1).int()[0, :].sum().item()
        else:  # encode up to n_steps_encode, decode to end of training part
            train_mask = (b_mask == 2).int()
            nsteps_encode = b_nsteps_encode[0]

        encode_x = b_ts_mat[:, shift:shift + nsteps_encode]
        decode_x = b_ts_mat[:, shift+nsteps_encode:shift+nsteps_encode+nsteps_decode]
        feat_len = encode_x.shape[1] + decode_x.shape[1]

        reconstr_x, pred_x, loss_func = net(encode_x, b_features[:, :feat_len],
                                            nsteps_encode, nsteps_decode,
                                            mask=train_mask[:, shift:shift+nsteps_encode],
                                            n_elbo_samp=n_elbo_samp,
                                            elbo_type=elbo_type,
                                            noise_std=noise_std)        
        assert(reconstr_x.shape[0] == n_elbo_samp)
        assert(reconstr_x.shape[2] == nsteps_encode - 1)
        assert(pred_x.shape[0] == n_elbo_samp)
        assert(pred_x.shape[2] == nsteps_decode)

        encode_x_shifted = encode_x[:, 1:, :]

        dec_loss = loss_func(decode_x, pred_x)
        enc_loss = loss_func(encode_x_shifted, reconstr_x)

        loss = torch.mean(dec_loss) + reconstr_weight * torch.mean(enc_loss)
        total_train_loss += loss.item() * bsize
        loss.backward()

        if clip_value is not None:
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value)

        optimizer.step()

        train_pred_mse = nn.MSELoss(reduction='sum')(pred_x, torch.stack([decode_x] * n_elbo_samp)).item()
        total_train_pred_loss += train_pred_mse
        b_num_train_pred_elements = torch.numel(pred_x)
        num_train_pred_elements += b_num_train_pred_elements

        reconstr_mask = train_mask[:, :nsteps_encode]
        train_reconstr_mse = criterion_func(torch.stack([reconstr_mask[:, 1:]] * n_elbo_samp),
                                            torch.stack([encode_x_shifted] * n_elbo_samp),
                                            reconstr_x).item()
        total_train_reconstr_loss += train_reconstr_mse
        b_num_train_reconstr_elements = reconstr_mask[:, 1:].sum().item() * reconstr_x.shape[-1] * n_elbo_samp
        num_train_reconstr_elements += b_num_train_reconstr_elements

        all_actual_encode.append(encode_x.detach().cpu().numpy())
        all_actual_decode.append(decode_x.detach().cpu().numpy())
        all_pred.append(pred_x.detach().cpu().numpy())
        all_reconstr.append(reconstr_x.detach().cpu().numpy())
        all_country_ids.append(b_country_id)

    return {
        'nsteps_decode': nsteps_decode,
        'nsteps_encode': nsteps_encode.item(),
        'train_pred_mse': total_train_pred_loss/num_train_pred_elements,
        'train_reconstr_mse': total_train_reconstr_loss/num_train_reconstr_elements,
        'train_actual_encode': np.concatenate(all_actual_encode, axis = -3),
        'train_actual_decode': np.concatenate(all_actual_decode, axis = -3),
        'train_reconstr': np.concatenate(all_reconstr, axis = -3), # n_trajectories, n_countries, n_times, n_states
        'train_pred': np.concatenate(all_pred, axis = -3),
        'train_loss':  total_train_loss/sum(map(len, all_country_ids)),
        'country_ids': torch.cat(all_country_ids).cpu().numpy()
    }

def predict(loader, net, device, elbo_type, n_elbo_samp, noise_std, shift=0):
    net.eval()
    all_actual_encode, all_actual_decode, all_reconstr, all_pred, all_country_ids = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            b_country_id, b_ts_mat, b_mask, b_features, b_nsteps_encode, b_nsteps_decode = batch

            b_ts_mat = b_ts_mat.float().to(device)
            b_features = b_features.float().to(device)
            b_mask = b_mask.to(device)

            pred_mask = (b_mask == 1).int()
            encode_mask = (b_mask == 2).int()

            nsteps_encode = (pred_mask[0, :] == 0).sum().item()
            nsteps_decode = pred_mask[0, :].sum().item()

            encode_x = b_ts_mat[:, shift:shift + nsteps_encode]
            decode_x = b_ts_mat[:, shift+nsteps_encode:shift+nsteps_encode+nsteps_decode]
            feat_len = encode_x.shape[1] + decode_x.shape[1]

            nsteps_decode -= (nsteps_encode+nsteps_decode - feat_len)

            reconstr_x, pred_x, _ = net(encode_x, b_features[:, :feat_len],
                                        nsteps_encode, nsteps_decode,
                                        mask=encode_mask[:, shift:shift+nsteps_encode],
                                        elbo_type=elbo_type, noise_std=noise_std,
                                        n_elbo_samp=n_elbo_samp)

            all_actual_encode.append(encode_x.detach().cpu().numpy())
            all_actual_decode.append(decode_x.detach().cpu().numpy())
            all_pred.append(pred_x.detach().cpu().numpy())
            all_reconstr.append(reconstr_x.detach().cpu().numpy())
            all_country_ids.append(b_country_id)

    metrs =  {
        'nsteps_decode': nsteps_decode,
        'nsteps_encode': nsteps_encode,
        'actual_encode': np.concatenate(all_actual_encode, axis = -3),
        'actual_decode': np.concatenate(all_actual_decode, axis = -3),
        'reconstr': np.concatenate(all_reconstr, axis = -3), # n_trajectories, n_countries, n_times, n_states
        'pred': np.concatenate(all_pred, axis = -3),
        'country_ids': torch.cat(all_country_ids).cpu().numpy(),
    }

    metrs['val_mse'] = mse(metrs['pred'], np.stack([metrs['actual_decode']] * n_elbo_samp))

    return metrs
