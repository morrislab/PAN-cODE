import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class AugmentedCovidDataset(Dataset):
    def __init__(self, df, unique_countries, first_cutoff, second_cutoff, feature_names, known_states = ['delI_smoothed', 'delD_smoothed'], target_type = 'count', nsteps_decode = 14, trunc = None):
        """
        Dataset of length number of countries; returns train/val splits by time
            as well as encode/pred split within the training times, with length of pred portion nsteps_decode

        If used during training time:
            - first_cutoff is the val_cutoff (i.e. start to first_cutoff is used as training)
            - second_cutoff is the test_cutoff (i.e. first_cutoff to second_cutoff is used for validation)

        If used for forecasting:
            - first_cutoff is the test_cutoff (i.e. start to test_cutoff is used as encoding)
            - second_cutoff is the date to forecast until (ex: test_cutoff + 4 weeks)
        """
        self.df = df
        if 'key' in self.df.columns:
            self.df = self.df.set_index('key')
        self.first_cutoff = first_cutoff       
        self.second_cutoff = second_cutoff
        # only take countries that have training data; dates before unseen cutoff
        self.df = self.df[(self.df.zero_time < first_cutoff) & (self.df['date'] < second_cutoff)]
        self.max_t = int(self.df.t.max())
        self.unique_countries = unique_countries
        self.num_countries = len(self.unique_countries)
        self.feature_names = feature_names
        self.known_states = known_states
        self.target_type = target_type
        self.nsteps_decode = nsteps_decode
        self.trunc = trunc # length of train encode to truncate to

    def __len__(self):
        return self.num_countries

    def prepend_false(self, arr, max_len):
        # appends false to the start of an array so it's the same shape as the mask
        # truncates from front if necessary
        if isinstance(arr, pd.Series):
            arr = arr.values
        if max_len >= len(arr):
            arr = np.append([False] * (max_len - len(arr)), arr)
        else:
            arr = arr[len(arr) - max_len:]
        return arr

    def get_country_df(self, idx):
        country_name = self.unique_countries[idx]
        country_df = self.df.loc[country_name].sort_values(by='t')
        country_id = country_df['country_id'].iloc[0]

        return country_id, country_df

    def get_ts_from_country(self, country_df):
        max_traj_len = self.max_t + 1
        n_ts_observed = max_traj_len - len(country_df)

        ts_mat = torch.zeros(max_traj_len, len(self.known_states), dtype=torch.float32)

        observed_ts = country_df[self.known_states].values
        ts_mat[n_ts_observed:] = torch.from_numpy(observed_ts[:max_traj_len, :]).float()

        return ts_mat

    def get_features_from_country(self, country_df):
        features = country_df[self.feature_names].values

        # Pads feature to same shape as ts_mat with first observed feature
        pad_feature = features[0, np.newaxis]
        pad_features = np.repeat(pad_feature, self.max_t + 1 - features.shape[0], axis=0)

        features = np.concatenate((pad_features, features))
        return features

    def get_mask(self, country_df):
        """
        Generates mask. Assigns training data 2, validation data 1, and test data -1.
        """
        mask = torch.zeros(self.max_t + 1, dtype=torch.float32)

        dates = country_df['date']

        first_ind = dates < self.first_cutoff
        sec_ind = (dates >= self.first_cutoff) & (dates < self.second_cutoff)

        mask[np.where(self.prepend_false(first_ind, self.max_t + 1))] = 2
        mask[np.where(self.prepend_false(sec_ind, self.max_t + 1))] = 1

        return mask

    def __getitem__(self, idx):
        """
        returns the following:
           - country_id (int)
           - ts_mat: 2D matrix of t by len(states)
           - mask: 1D mask (for train/val/split) of length t
                   value of 2 for t < first_cutoff (i.e. training)
                   value of 1 for second_cutoff > t >= first_cutoff (i.e. val)
                   value of 0 for unknown data (should only appear at front)  
           - features: 2D matrix of t by len(feature_names)
           - nsteps_encode, nsteps_decode: number of time steps in the encode and pred portion
        """
        country_id, country_df = self.get_country_df(idx)
        ts_mat = self.get_ts_from_country(country_df)
        features = self.get_features_from_country(country_df)
        mask = self.get_mask(country_df)

        if self.target_type == 'log':
            m1 = (ts_mat > 0)
            ts_mat[m1] = np.log(ts_mat[m1])
        elif self.target_type == 'shifted_log':
            m1 = (ts_mat > 0)
            ts_mat[m1] = np.log(ts_mat[m1] + 1)
        elif self.target_type == 'count':
            pass
        else:
            raise NotImplementedError

        nsteps_encode = np.isin(mask, [2, 0]).sum() - self.nsteps_decode

        if self.trunc is not None:
            trunc_index = max(0, nsteps_encode - self.trunc)
            ts_mat = ts_mat[trunc_index:, :]
            mask = mask[trunc_index:]
            features = features[trunc_index:, :]
            nsteps_encode = min(self.trunc, nsteps_encode)

        return country_id, ts_mat, mask, features, nsteps_encode, self.nsteps_decode


class SampledCovidDataset(AugmentedCovidDataset):
    """
    Dataloader which randomly selects position from which to begin decode. This class should be
    used in conjunction with the conditional Latent ODE to expose the network to additional
    conditional values.
    """

    def __init__(self, df, unique_countries, first_cutoff, second_cutoff, feature_names,
                 known_states=['delI_smoothed', 'delD_smoothed'], target_type='count',
                 nsteps_decode=None, trunc=None, samp_pad=(0, 0)):
        """
        Args:
            samp_pad (tuple of int): Left / right padding added to sample bounds.
        """
        super().__init__(df, unique_countries, first_cutoff, second_cutoff, feature_names,
                         known_states, target_type, nsteps_decode, trunc)

        ds_first_obs = self.get_fully_obs_ind(unique_countries)
        ds_enc_len = self.get_enc_len()
        lower = samp_pad[1]
        upper = ds_enc_len - ds_first_obs - samp_pad[0]
        if upper < lower:
            upper = lower

        self.nsteps_decode = np.random.randint(lower, upper)

    def get_fully_obs_ind(self, unique_countries):
        """
        Returns the first index where all time series contain an observation.
        """
        ds_first_obs = 0
        for idx in range(len(unique_countries)):
            _, country_df = self.get_country_df(idx)
            mask = self.get_mask(country_df).numpy()

            first_obs = np.where(mask == 2)[0][0]
            ds_first_obs = max(ds_first_obs, first_obs)
        return ds_first_obs

    def get_enc_len(self):
        """
        Returns the length of trajectory to first cutoff.
        """
        _, country_df = self.get_country_df(0)
        mask = self.get_mask(country_df).numpy()

        ds_enc_len = np.isin(mask, [2, 0]).sum()
        return ds_enc_len
