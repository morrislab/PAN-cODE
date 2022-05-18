'''
Sample usage: 
python summarize.py --experiment_name LatentODEBasicFourWeek \
    --data_dir /scratch/ssd001/home/haoran/projects/CovidForecast/data \
        --output_dir /scratch/hdd001/home/haoran/CovidForecast 

Outputs in the experiment folder:
- For each imputation strategy:
    - '{forecast_date}-{args.experiment_name}_{strategy}.csv': weekly aggregated predictions in Covid-19 Forecast Hub format
    - '{forecast_date}-{args.experiment_name}_{strategy}.pkl': pickled dictionary of dataframes with all trajectories
- model_summary.csv: a summary of the models, sorted by select_metric
'''

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
import pickle
import os
import sys
sys.path.append(os.getcwd())

from tqdm.auto import tqdm
from tqdm import trange

import json
import copy
import random

from lib.Constants import *
from lib.Dataloader import AugmentedCovidDataset
from lib.utils import plot_combo

from lib.models import criterion_func, load_model
from argparse import ArgumentParser
from lib.train_helper import predict
from lib.utils import predictions_to_df, load_data_features, impute_df, strict_policy, loose_policy
import shutil

parser = ArgumentParser(description = '''Outputs a csv file with projections in the format \
    specified by https://github.com/reichlab/covid19-forecast-hub/tree/master/data-processed. \
    Note that the forecast_date will be a Monday, while forecasts will be made weekly from Sunday-Saturday, \
    so the first week forecast will only be from Monday-Saturday.''')
parser.add_argument('--experiment_name', type=str, required = True)
parser.add_argument('--data_dir', type=Path, required=True)
parser.add_argument('--date_cutoff', type=str, default = None, 
        help = 'date to encode up to but not include; should be a Monday; will infer from training argparse by default')
parser.add_argument('--date_cutoff_interventions', type = str, default = None,
        help = "Date to stop observing interventions and start imputing last intervention. Default is to use all available data.")
parser.add_argument('--n_weeks_ahead', type=int, default = 8, 
        help = 'forecast deaths and incidents for 1...x weeks after date_cutoff.')
parser.add_argument('--output_dir', type = Path, required = True)
parser.add_argument('--n_trajectories', type = int, default = 50)
parser.add_argument('--select_metric', type = str, choices = ['all_mse','all_mae'], default = 'all_mae')
parser.add_argument('--covid_hub_dir', type=Path, help = "should point to the data-processed folder")
parser.add_argument('--us_sum_states', action = 'store_true', help = "compute US projection as sum of all states")
parser.add_argument('--states_sum_counties', action = 'store_true', help = "compute state projections as sum of all counties")
args = parser.parse_args()

exp_root = Path(args.output_dir)/args.experiment_name

temp = []
for i in exp_root.glob('**/val_metrics.json'):
    train_args = json.load((i.parent/'args.json').open('r'))
    val_metrics = json.load((i).open('r'))
    temp.append({**train_args, **val_metrics})

assert(len(temp) > 0), "No completed experiments!"
df_models = pd.DataFrame(temp).sort_values(by = args.select_metric, ascending = True)
df_models.to_csv(exp_root/'models_summary.csv')

best_model_path = Path(df_models.iloc[0]['output_dir'])
shutil.copy(best_model_path/'args.json', exp_root/'best_args.json')

print(df_models.head())

assert(len(df_models['date_cutoff'].unique()) == 1)
date_cutoff = pd.Timestamp(args.date_cutoff) if args.date_cutoff is not None else pd.Timestamp(df_models.iloc[0]['date_cutoff'])
final_date = date_cutoff + pd.Timedelta(days = args.n_weeks_ahead * 7 - 2)
assert(final_date.strftime('%A') == 'Saturday')

train_args = df_models.iloc[0].to_dict()

if not args.states_sum_counties:
    train_args['include_counties'] = False

all_features, df, countries, id_mapping, reverse_id_mapping = load_data_features(args.data_dir, train_args, override_data_type=True)

unique_countries = df.key.unique()

if train_args['ohe_features']:
    all_features += unique_countries.tolist()
    for country in unique_countries:
        df[country] = (df['key'] == country).astype(int)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = load_model(all_features, train_args)
net.load_state_dict(torch.load(best_model_path/'model', map_location=torch.device('cpu')))
net = net.to(device)

# we impute by some extra days just in case
imputed_dfs = {
    'impute_only_when_missing': impute_df(df, final_date + pd.Timedelta(days = 20), all_features, start_date = None, impute_strategy = 'last'),
    'impute_from_cutoff': impute_df(df, final_date + pd.Timedelta(days = 20), all_features, start_date = date_cutoff, impute_strategy = 'cutoff'),
    'all_interventions': impute_df(df, final_date + pd.Timedelta(days = 20), all_features, start_date = date_cutoff, impute_strategy = strict_policy),
    'no_interventions': impute_df(df, final_date + pd.Timedelta(days = 20), all_features, start_date = date_cutoff, impute_strategy = loose_policy),
}

for key, imputed_df in imputed_dfs.items():
    test_dataset = AugmentedCovidDataset(imputed_df, unique_countries, date_cutoff, final_date + pd.Timedelta(days = 1), all_features,
                                    known_states = ['delI_smoothed', 'delD_smoothed'] if train_args['smoothed'] else ['delI', 'delD'],
                                    target_type = train_args['target_type'],
                                    nsteps_decode = 1,
                                                trunc = None)
    test_loader = DataLoader(test_dataset, batch_size = int(train_args['batch_size']*2), shuffle = False)

    predictions = predict(test_loader, net, device, n_elbo_samp=args.n_trajectories, noise_std=train_args['noise_std'],
                        elbo_type=train_args['elbo_type'])
    pred_dfs, test_metrics = predictions_to_df(predictions, df, date_cutoff, train_args['target_type'])    

    all_states = np.unique(pred_dfs['mean'][(pred_dfs['mean'].key.str.startswith('US_')) & (pred_dfs['mean'].aggregation_level == 1)].key.values)
    if args.states_sum_counties:
        for i in pred_dfs:
            for state in all_states:
                temp = pred_dfs[i]
                temp = temp[(temp.key.str.startswith(state)) & (temp.aggregation_level == 2)].groupby('date').agg({
                    'delI_pred': 'sum',
                    'delD_pred': 'sum'
                })
                pred_dfs[i].loc[pred_dfs[i].key == state, ['delI_pred', 'delD_pred']] = temp.values

    if args.us_sum_states:
        for i in pred_dfs:
            temp = pred_dfs[i]
            temp = temp[temp.key.isin(all_states)].groupby('date').agg({
                'delI_pred': 'sum',
                'delD_pred': 'sum'
            })
            pred_dfs[i].loc[pred_dfs[i].key == 'US', ['delI_pred', 'delD_pred']] = temp.values            
    
    pickle.dump(pred_dfs, (exp_root/f'{str(date_cutoff.date())}-{args.experiment_name}-{key}.pkl').open('wb'))
    preds = pred_dfs['mean']

    if key in ['impute_only_when_missing', 'impute_from_cutoff']:
        preds = preds[~pd.isnull(preds.covid_hub_id)]
        ## start is always Sunday, end is always Saturday
        start = date_cutoff - pd.Timedelta(days = 1)
        end = start + pd.Timedelta(days = 6)
        dfs_raw = []
        for i in range(args.n_weeks_ahead):
            pred_i = (preds[(preds.date >= start) & (preds.date <= end)]
                        .groupby('covid_hub_id')
                        .agg({'delI_pred': 'sum', 'delD_pred': 'sum'})
                        .reset_index()
                        .rename(columns = {'delI_pred': f'{i+1} wk ahead inc case', 'delD_pred': f'{i+1} wk ahead inc death'})
                        .melt(id_vars = ['covid_hub_id'], value_vars = [ f'{i+1} wk ahead inc case', f'{i+1} wk ahead inc death'])
                        .rename(columns = {'variable': 'target', 'covid_hub_id': 'location'}))
            pred_i['quantile'] = "NA"
            pred_i['type'] = 'point'
            pred_i['target_end_date'] = str(end.date())
            pred_i['forecast_date'] = str(date_cutoff.date())
            
            dfs_raw.append(pred_i)

            start += pd.Timedelta(days = 7)
            end += pd.Timedelta(days = 7)

        df_all = pd.concat(dfs_raw, ignore_index = True)[["forecast_date","target","target_end_date","location","type","quantile","value"]]
        df_all.to_csv(exp_root/f'{str(date_cutoff.date())}-{args.experiment_name}-{key}.csv', index = False)

        if key == 'impute_from_cutoff' and args.covid_hub_dir is not None:
            (args.covid_hub_dir/args.experiment_name).mkdir(parents = True, exist_ok = True)
            df_all.to_csv(args.covid_hub_dir/args.experiment_name/f'{str(date_cutoff.date())}-{args.experiment_name}.csv', index = False)
            