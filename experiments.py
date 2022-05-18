import numpy as np
from lib import Constants
import pandas as pd
from itertools import product

def combinations(grid):
    return list(dict(zip(grid.keys(), values)) for values in product(*grid.values()))

def get_script_name(experiment):
    if experiment not in globals():
        raise NotImplementedError(experiment)
    return globals()[experiment].fname

def add_common_args(parser):
    parser.add_argument('--model', type = str, choices =  ['lode', 'gruode', 'gru', 'vgru'], required = True)
    parser.add_argument('--data_type', type = str, choices =  ['all', 'debug', 'US'], required = True)
    parser.add_argument('--randomize_training', action = 'store_true')
    parser.add_argument('--concat_cond_ts', action = 'store_true', help = 'concatenate interventions as time series to z instead of statically to z0. \
                    Affects lode and vgru.')
    return parser


#### Experiment Bases
class ExpBase:
    def get_hparams(self):
        return combinations(self.hparams)

class LatentODEBase(ExpBase):
    fname = "train.py"
    def __init__(self):
        grid = {
            'model': ['lode'],
            'latent_dim': [32],
            'n_layer': [2, 3],
            'n_units': [32, 64],
            'feature_set': ['condensed'],
            'final_activation': ['none'],
            'target_type': ['log'],
            'reconstr_weight': [0.0, 1.0],
            'ohe_features': [False],
            'lr': [1e-3],
            'batch_size': [1024], # full batch training
            'cond_features': ['all', 'stringency', 'none'],
            'smoothed': [True]
        }
        self.hparams = { **self.hparams,  **grid} if 'hparams' in self.__dict__ else grid

class VGRUBase(ExpBase):
    fname = "train.py"
    def __init__(self):
        grid = {
            'model': ['vgru'],
            'latent_dim': [8, 32],
            'n_layer': [2, 3],
            'n_units': [32, 64],
            'feature_set': ['condensed'],
            'final_activation': ['none'],
            'target_type': ['shifted_log'],
            'reconstr_weight': [0.0, 1.0],
            'ohe_features': [False],
            'lr': [1e-3],
            'batch_size': [256], 
            'dropout_p': [0.0, 0.25],
            'cond_features': ['stringency'],
            'smoothed': [True],
            'include_counties': [True],
        }
        self.hparams = { **self.hparams,  **grid} if 'hparams' in self.__dict__ else grid

class GRUBase(ExpBase):
    fname = "train.py"
    def __init__(self):
        grid = {
            'model': ['gru'],
            'latent_dim': [8, 16, 32],
            'n_units': [32, 64],
            'feature_set': ['condensed'],
            'final_activation': ['none'],
            'target_type': ['shifted_log'],
            'reconstr_weight': [0.0],
            'ohe_features': [False],
            'lr': [1e-3],
            'batch_size': [256], 
            'cond_features': ['stringency'],
            'smoothed': [True],
            'include_counties': [True],
        }
        self.hparams = { **self.hparams,  **grid} if 'hparams' in self.__dict__ else grid        


class GRUODEBase(ExpBase):
    fname = "train.py"
    def __init__(self):
        grid = {
            'model': ['gruode'],
            'latent_dim': [8, 32],
            'n_layer': [2, 3],
            'n_units': [32, 64],
            'feature_set': ['condensed'],
            'final_activation': ['none'],
            'target_type': ['shifted_log'],
            'reconstr_weight': [0.0],
            'ohe_features': [False],
            'lr': [1e-3],
            'batch_size': [256], 
            'cond_features': ['stringency'],
            'smoothed': [True],
            'include_counties': [True],
        }
        self.hparams = { **self.hparams,  **grid} if 'hparams' in self.__dict__ else grid       

class WeekBase(ExpBase):
    def __init__(self, ndays, anchor_date):
        grid = {
            'date_cutoff': [str(pd.Timestamp(anchor_date))],
            'n_val_days': [ndays]
        }
        self.hparams = { **self.hparams,  **grid} if 'hparams' in self.__dict__ else grid

#### write experiments here
def get_exp_name(args):
    return args.model + '_' + str(args.anchor_date) + '_' + str(args.n_weeks_ahead) + '_' + str(args.data_type) + ('_randomize' if args.randomize_training else '') \
                    + ('_cond_ts' if args.concat_cond_ts else '')

def get_hparams(args):
    model_bases = {
        'vgru': VGRUBase,
        'lode': LatentODEBase,
        'gru': GRUBase,
        'gruode': GRUODEBase
    }
    model_base_class = model_bases[args.model]

    class Experiment(model_base_class, WeekBase):
        def __init__(self):
            model_base_class.__init__(self)
            WeekBase.__init__(self, args.n_weeks_ahead * 7, args.anchor_date)

            grid = {
                'experiment_name': [get_exp_name(args)],
                'data_type': [args.data_type],
                'randomize_training': [args.randomize_training],
                'concat_cond_ts': [args.concat_cond_ts]                
            }
            self.hparams = { **self.hparams,  **grid} if 'hparams' in self.__dict__ else grid

    experiment = Experiment()
    return experiment.get_hparams()



# class LatentODEBasicFourWeek(LatentODEBase, FourWeek):
#     def __init__(self):
#         self.hparams = {
#             'data_type': ['US'],
#             'smoothed': [True],
#         }
#         LatentODEBase.__init__(self)
#         FourWeek.__init__(self)

# class RandomizedCondIWLatentODEFourWeek(LatentODEBase, FourWeek):
#     def __init__(self):
#         self.hparams = {
#             'data_type': ['US'],
#             'smoothed': [True],
#             'noise_std': [1, 0.1],
#             'cond_inds': ['[-1]'],
#             'elbo_type': ['iwae'],
#             'randomize_training': [True]
#         }
#         LatentODEBase.__init__(self)
#         FourWeek.__init__(self)

# class LatentODEVAEHyperGrid(LatentODEBase, FourWeek):
#     def __init__(self):
#         self.hparams = {
#             'data_type': ['US'],
#             'smoothed': [True],
#             'n_train_trajectories': [10, 25, 50],
#         }
#         LatentODEBase.__init__(self)
#         FourWeek.__init__(self)
