import torch
import os
from pathlib import Path
import getpass
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from lib.train_helper import mse, mae
import pandas as pd
import json
import pickle

class EarlyStopping:
    # adapted from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    def __init__(self, patience=7):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, path, net):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            save_model(path, net)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            save_model(path, net)
            self.counter = 0

def save_model(path, net): # saves the best model so far for early stopping
    torch.save(net.state_dict(), path)

def save_checkpoint(model, optimizer, scheduler, start_step, es, rng, rng_python, logs):  # saves checkpoint in case of preemption
    slurm_job_id = os.environ.get('SLURM_JOB_ID')

    if slurm_job_id is not None and Path(f'/checkpoint/{getpass.getuser()}/{slurm_job_id}/').exists():
        torch.save({'model_dict': model.state_dict(),
                    'optimizer_dict': optimizer.state_dict(),
                    'scheduler_dict': scheduler.state_dict(),
                    'start_step': start_step,
                    'es': es,
                    'rng': rng,
                    'rng_python': rng_python,
                    'logs': logs
        }
                   ,
                   Path(f'/checkpoint/{getpass.getuser()}/{slurm_job_id}/chkpt').open('wb')
                  )


def has_checkpoint():
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    if slurm_job_id is not None and Path(f'/checkpoint/{getpass.getuser()}/{slurm_job_id}/chkpt').exists():
        return True
    return False

def load_checkpoint():
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    fname = Path(f'/checkpoint/{getpass.getuser()}/{slurm_job_id}/chkpt')
    if slurm_job_id is not None and fname.exists():
        return torch.load(fname)

class Exp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)

def plot_country_ax(predict_dict, idx, ax, exp = False, plot_infected = True, plot_deaths = True):
    transform = (lambda x: np.exp(x)) if exp else (lambda x: x)
    actual = np.concatenate([predict_dict['actual_encode'], predict_dict['actual_decode']], axis = -2)
    pred = np.concatenate([predict_dict['reconstr'], predict_dict['pred']], axis = -2)
    x = np.arange(1, actual.shape[1]+1)
    if plot_infected:
        ax.plot(x, transform(actual[idx, :, 0]), 'b.', label = 'Infected')
    if plot_deaths:
        ax.plot(x, transform(actual[idx, :, 1]), 'r.', label = 'Dead')

    if pred.ndim == 4:
        n_trajectories = pred.shape[0]
        for i in range(n_trajectories): # each trajectory
            if plot_infected:
                ax.plot(x, transform(pred[i, idx, :, 0]), 'b-', alpha = 0.2)
            if plot_deaths:
                ax.plot(x, transform(pred[i, idx, :, 1]), 'r-', alpha = 0.2)
        ax.set_title("MSE Infected = %.2e, MSE Dead = %.2e" % (mse(transform(predict_dict['pred'][:, idx, :, 0]), np.stack([transform(predict_dict['actual_decode'][idx, :, 0])] * n_trajectories)),
                                        mse(transform(predict_dict['pred'][:, idx, :, 1]), np.stack([transform(predict_dict['actual_decode'][idx, :, 1])] * n_trajectories))))
    elif pred.ndim == 3:
        if plot_infected:
            ax.plot(x, transform(pred[idx, :, 0]), 'b-')
        if plot_deaths:
            ax.plot(x, transform(pred[idx, :, 1]), 'r-')
        ax.set_title("MSE Infected = %.2e, MSE Dead = %.2e" % (mse(transform(predict_dict['pred'][idx, :, 0]), transform(predict_dict['actual_decode'][idx, :, 0])),
                                    mse(transform(predict_dict['pred'][idx, :, 1]), transform(predict_dict['actual_decode'][idx, :, 1]))))

    colors = ['C0', 'y', 'g']
    line = [predict_dict['actual_encode'].shape[-2]]
    shade_indices = [1] + line + [x.max()]
    for i in range(0, len(shade_indices)-1):
        color = colors[i]
        ax.axvspan(shade_indices[i], shade_indices[i+1], alpha = 0.1, color = color)
    ax.set_xlim(left = x.min(), right = x.max())

    return ax

def load_data_features(data_dir, arg_dict, override_data_type = False):
    feature_json = json.load((data_dir / 'features.json').open('r'))
    condensed_ts = feature_json['condensed_ts'] + feature_json['weather_feature']
    expanded_ts = feature_json['all_ts'] + feature_json['weather_feature']
    static_features = feature_json['basic_features']
    if arg_dict['feature_set'] == 'states':
        all_features = []
    elif arg_dict['feature_set'] == 'condensed':
        all_features = condensed_ts + static_features
    elif arg_dict['feature_set'] == 'expanded':
        all_features = expanded_ts + static_features

    processed_data = pickle.load((data_dir / 'processed_data.pkl').open('rb'))
    countries = processed_data['countries']
    id_mapping = processed_data['id_mapping']
    reverse_id_mapping = processed_data['reverse_id_mapping']
    df = processed_data['df']

    if arg_dict['data_type'] == 'debug':
        if not override_data_type:
            df = df[df.country_name.isin(['Canada', 'United States of America'])]
    elif arg_dict['data_type'] == 'US':
        if not override_data_type:
            df = df[df.key.str.startswith('US')]
        all_features = [i for i in all_features if i not in static_features]

    if not arg_dict['include_counties']:
        df = df[df.aggregation_level <= 1]

    return all_features, df, countries, id_mapping, reverse_id_mapping

def plot_combo(predict_dict, idx):
    fig, axs = plt.subplots(ncols = 2, nrows = 2, figsize = (10, 9))
    plot_country_ax(predict_dict, idx, axs[0][0], exp = False, plot_infected = True, plot_deaths = False)
    plot_country_ax(predict_dict, idx, axs[0][1], exp = False, plot_infected = False, plot_deaths = True)
    plot_country_ax(predict_dict, idx, axs[1][0], exp = True, plot_infected = True, plot_deaths = False)
    plot_country_ax(predict_dict, idx, axs[1][1], exp = True, plot_infected = False, plot_deaths = True)
    return fig

def predictions_to_df(pred_output, df, start_date, target_type):
    dfs = {}
    start_date = pd.Timestamp(start_date)
    n_trajectories = pred_output['pred'].shape[0]    # n_trajectories, n_countries, n_days, n_states
    n_days_forecast = pred_output['pred'].shape[2]
    if target_type == 'log':
        pred_output['pred'] = np.clip(np.exp(pred_output['pred']), 0, None)
    elif target_type == 'shifted_log':
        pred_output['pred'] = np.clip(np.exp(pred_output['pred']) - 1, 0, None)
    mean_traj = np.mean(pred_output['pred'], axis = 0, keepdims = True)

    # concatenate mean trajectory to matrix for convenience
    pred_output['pred'] = np.concatenate((pred_output['pred'], mean_traj), axis = 0)

    for i in range(n_trajectories + 1):
        countries_raw = []
        for c, country_id in enumerate(pred_output['country_ids']):
            countries_raw.append(
                pd.DataFrame(
                    {
                        'country_id': country_id,
                        'delI_pred': pred_output['pred'][i, c, :, 0],
                        'delD_pred': pred_output['pred'][i, c, :, 1],
                        'date': [start_date + pd.Timedelta(days = j) for j in range(n_days_forecast)]
                    }
                )
            )
        
        dfs[i] = (pd.concat(countries_raw , ignore_index=True)
                        .reset_index(drop = True)
                        .merge(df, on = ('date', 'country_id'), how = 'inner')
                        .sort_values(by = ['country_id', 'date']))

    dfs['mean'] = dfs.pop(n_trajectories)
    metrics = {
        'I_mse': mse(dfs['mean']['delI_pred'], dfs['mean']['delI']),
        'I_mae': mae(dfs['mean']['delI_pred'], dfs['mean']['delI']),
        'D_mse': mse(dfs['mean']['delD_pred'], dfs['mean']['delD']),
        'D_mae': mae(dfs['mean']['delD_pred'], dfs['mean']['delD']),
        'all_mse': mse(dfs['mean'][['delI_pred', 'delD_pred']].values, dfs['mean'][['delI', 'delD']].values),
        'all_mae': mae(dfs['mean'][['delI_pred', 'delD_pred']].values, dfs['mean'][['delI', 'delD']].values)
    }
    return dfs, metrics

def impute_df(df, final_date, all_features, start_date = None, impute_strategy = 'last'):
    '''
    Imputes features in df up to and including final_date.
    If start_date is None, start imputing only the missing dates, else, replace all data starting from start_date.
    Impute_strategy: 
        - If "last", impute forward using the final entry in df
        - If "cutoff", impute forward the last entry directly prior to start_date
        - else, should be a function that takes in the last entry (as a pd Series) and returns a dictionary
    '''
    df = df.sort_values(by = ['key', 'date']).set_index('key')
    if start_date is not None:
        assert impute_strategy != 'last'
        df = df[df.date < start_date]

    last_vals = df.groupby('key').last()
    new_rows = []

    for country in df.index.unique():
        df_country = df.loc[country].reset_index()
        init_date = df_country['date'].max() + pd.Timedelta(days = 1)
        init_t = df_country['t'].max() + 1
        n_days_impute = (final_date - init_date).days

        new_rows.append(
            pd.DataFrame({**{
                'key': country,
                't': np.arange(init_t, init_t + n_days_impute),
                'date': [init_date  + pd.Timedelta(days = i) for i in range(n_days_impute)],
                'zero_time': last_vals.loc[country, 'zero_time'],
                'country_id': last_vals.loc[country, 'country_id'],
            }, **(last_vals.loc[country, all_features].to_dict() if impute_strategy in ['last', 'cutoff'] else impute_strategy(last_vals.loc[country, all_features].to_dict(), df_country))})
        )
        
    return pd.concat([df.reset_index(), pd.concat(new_rows, ignore_index = True)], ignore_index = True).reset_index(drop = True).sort_values(by = ['key', 'date'])

def strict_policy(template, df_country):
    return df_country.loc[df_country['stringency_index_norm'].idxmax(), list(template.keys())].to_dict()

def loose_policy(template, df_country):
    return df_country.loc[df_country['stringency_index_norm'].idxmin(), list(template.keys())].to_dict()

def get_cond_inds(all_features, args_dict):
    if args_dict['cond_features'] == 'none':
        return torch.Tensor([]).long()
    elif args_dict['cond_features'] == 'stringency':
        indices = ['stringency_index_norm', 'GovernmentResponseIndex_norm', 'ContainmentHealthIndex_norm', 'EconomicSupportIndex_norm']
        return torch.Tensor([all_features.index(i) for i in indices]).long()
    else:
        return torch.arange(len(all_features)).long()