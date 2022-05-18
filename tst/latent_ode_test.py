import unittest

import torch

from lib.latent_ode import AutoregressiveDecoder


class TestAutoregressiveDecoder(unittest.TestCase):
    batch_size = 2
    seq_len = 3
    in_dim = 4
    out_dim = 2
    feat_dim = 1

    def test_forward_base(self):
        dec = AutoregressiveDecoder(self.in_dim)
        dec.final.weight.data.fill_(1)

        true_out = self.out_dim * (self.in_dim + self.out_dim) + self.in_dim

        test_in = torch.ones(self.batch_size, self.seq_len, self.in_dim)
        test_init_state = torch.ones(self.batch_size, self.out_dim)

        test_out = dec.forward(test_init_state, test_in)

        self.assertTrue(torch.all(test_out[:, 0, :] == test_init_state))
        self.assertTrue(torch.all(test_out[:, -1, :] == true_out))

    def test_forward_all_states(self):
        dec = AutoregressiveDecoder(self.in_dim)
        dec.final.weight.data.fill_(1)

        test_in = torch.zeros(self.batch_size, self.seq_len, self.in_dim)
        test_init_state = torch.ones(self.batch_size, self.out_dim)

        test_all_states = torch.ones(self.batch_size, self.seq_len, self.out_dim)
        test_all_states[:, 1, :].fill_(3)

        test_out = dec.forward(test_init_state, test_in, test_all_states)

        self.assertTrue(torch.all(test_out[:, 0, :] == test_init_state))
        self.assertTrue(torch.all(test_out[:, 1, :] == 2))
        self.assertTrue(torch.all(test_out[:, 2, :] == 6))

    def test_forward_features(self):
        dec = AutoregressiveDecoder(self.in_dim + self.feat_dim)
        dec.final.weight.data.fill_(1)

        test_in = torch.zeros(self.batch_size, self.seq_len, self.in_dim)
        test_init_state = torch.ones(self.batch_size, self.out_dim)

        test_features = (torch.arange(self.seq_len) + 1).view(1, -1, 1)
        test_features = torch.cat([test_features] * self.batch_size, dim=0)
        test_out = dec.forward(test_init_state, test_in, features=test_features)

        self.assertTrue(torch.all(test_out[:, 0, :] == test_init_state))
        self.assertTrue(torch.all(test_out[:, 1, :] == 3))
        self.assertTrue(torch.all(test_out[:, 2, :] == 8))

    def test_forward_features_states(self):
        dec = AutoregressiveDecoder(self.in_dim + self.feat_dim)
        dec.final.weight.data.fill_(1)

        test_in = torch.zeros(self.batch_size, self.seq_len, self.in_dim)
        test_init_state = torch.ones(self.batch_size, self.out_dim)

        test_features = (torch.arange(self.seq_len) + 1).view(1, -1, 1)
        test_features = torch.cat([test_features] * self.batch_size, dim=0)

        test_all_states = (-torch.arange(self.seq_len) - 5).view(1, -1, 1)
        test_all_states = torch.cat([test_all_states] * self.batch_size, dim=0)
        test_all_states = torch.cat([test_all_states] * self.out_dim, dim=-1)

        test_out = dec.forward(test_init_state, test_in, test_all_states, test_features)

        self.assertTrue(torch.all(test_out[:, 0, :] == test_init_state))
        self.assertTrue(torch.all(test_out[:, 1, :] == 3))
        self.assertTrue(torch.all(test_out[:, 2, :] == -10))


if __name__ == '__main__':
    unittest.main()