import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
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
import socket

from lib.Constants import *
from lib.Dataloader import AugmentedCovidDataset, SampledCovidDataset
from lib.utils import EarlyStopping, plot_combo
from experiments import add_common_args

from lib.models import criterion_func, load_model
from argparse import ArgumentParser
from lib.train_helper import train_epoch, predict
from lib.utils import save_checkpoint, load_checkpoint, has_checkpoint, save_model, predictions_to_df, load_data_features

parser = ArgumentParser()
parser.add_argument('--experiment_name', type=str, default = '')
parser.add_argument('--data_dir', type=Path, required=True)
parser.add_argument('--date_cutoff', type=str, default = '2021-02-08', help = 'Should be a Monday')
parser.add_argument('--n_val_days', type=int, default = 30)
parser.add_argument('--feature_set', type = str, choices =  ['states', 'condensed', 'expanded'], default = 'condensed')
parser.add_argument('--output_dir', type = Path, required = True)
parser.add_argument('--batch_size', type=int, default = 32)
parser.add_argument('--smoothed', action = 'store_true')
parser.add_argument('--target_type', type = str, choices =  ['count', 'log', 'shifted_log'], default = 'log')
parser.add_argument('--lr', type = float, default = 1e-3)
parser.add_argument('--n_layer', type = int, default = 3)
parser.add_argument('--n_units', type = int, default = 50)
parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--latent_dim', type = int, default = 32)
parser.add_argument('--reconstr_weight', type = float, default = 1.0)
parser.add_argument('--final_activation', type = str, choices =  ['softplus', 'relu', 'none'], default = 'softplus')
parser.add_argument('--dropout_p', type = float, default = 0.0)
parser.add_argument('--ohe_features', action = 'store_true')
parser.add_argument('--min_epochs', type = int, default = 1000)
parser.add_argument('--trunc_patience', type = int, default = 20)
parser.add_argument('--n_train_trajectories', type = int, default = 10)
parser.add_argument('--n_finetune_epochs', type = int, default = 5)
parser.add_argument('--checkpoint_freq', type = int, default = 20)
parser.add_argument('--elbo_type', type = str, choices = ["vae", "iwae"], default = "vae")
parser.add_argument('--noise_std', type = float, default = 0.5)
parser.add_argument('--cond_features', type = str, choices = ['stringency', 'all', 'none'], default = 'stringency')
parser.add_argument('--clip_value', type = float)
parser.add_argument('--dec_type', type = str, choices = ["AR", "NN"], default="AR")
parser.add_argument('--stl', action = 'store_true')
parser.add_argument('--shift', type = int, default = 0)
parser.add_argument('--include_counties', action = 'store_true')
parser.add_argument('--num_workers', type = int, default = 4)
parser = add_common_args(parser)
args = parser.parse_args()

print("Environment:")
print("\tPython: {}".format(sys.version.split(" ")[0]))
print("\tPyTorch: {}".format(torch.__version__))
print("\tCUDA: {}".format(torch.version.cuda))
print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
print("\tNumPy: {}".format(np.__version__))
print("\tNode: {}".format(socket.gethostname()))

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

(args.output_dir).mkdir(parents=True, exist_ok=True)

(args.output_dir/'job_id').write_text(str(os.environ['SLURM_JOBID']))

min_decode_times = 2
max_decode_times = 60

with (args.output_dir/'args.json').open('w') as f:
    temp = copy.deepcopy(vars(args))
    for i in temp:
        if isinstance(temp[i], Path):
            temp[i] = os.fspath(temp[i])
    json.dump(temp, f, indent = 4)
    print(json.dumps(temp, indent = 4, sort_keys = True))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

all_features, df, countries, id_mapping, reverse_id_mapping = load_data_features(args.data_dir, vars(args))

test_cutoff = pd.Timestamp(args.date_cutoff)
assert(test_cutoff.strftime('%A') == 'Monday')
val_cutoff = test_cutoff - pd.Timedelta(days = args.n_val_days)

print("Test cutoff: %s\nValidation cutoff: %s" % (test_cutoff, val_cutoff))

train_val_df = df[(df.date < test_cutoff) & (df.zero_time < val_cutoff)]
unique_countries = train_val_df.key.unique()
print("# samples: " + str(len(unique_countries)))

if args.ohe_features:
    all_features += unique_countries.tolist()
    for country in unique_countries:
        train_val_df[country] = (train_val_df['key'] == country).astype(int)
        df[country] = (df['key'] == country).astype(int)

if args.model == 'lode' and args.concat_cond_ts:
    assert(args.cond_features != 'none')

net = load_model(all_features, vars(args)).to(device)
print("# parameters: ", sum(a.numel() for a in net.parameters()))

optimizer = Adam(params = net.parameters(), lr = args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 10)
es = EarlyStopping(patience=20)

if args.data_type == 'debug':
    n_epochs = 30
else:
    n_epochs = 10000
es_path = args.output_dir / 'model'

if has_checkpoint() and not args.data_type == 'debug':
    state = load_checkpoint()
    net.load_state_dict(state['model_dict'])
    optimizer.load_state_dict(state['optimizer_dict'])
    scheduler.load_state_dict(state['scheduler_dict'])
    start_step = state['start_step']
    es = state['es']
    torch.random.set_rng_state(state['rng'])
    random.setstate(state['rng_python'])
    logs = state['logs']
    print("Loaded checkpoint at epoch %s" % start_step, flush = True)
else:
    start_step = 1
    logs = []

known_states = ['delI_smoothed', 'delD_smoothed'] if args.smoothed else ['delI', 'delD']

for epoch in range(start_step, n_epochs+1):
    if es.early_stop:
        break

    if args.randomize_training:
        trunc = 5 + epoch//args.trunc_patience

        nsteps_decode = random.randint(min_decode_times,
                                       min_decode_times + min(max_decode_times - min_decode_times, epoch//args.trunc_patience))

        train_val_dataset = SampledCovidDataset(train_val_df, unique_countries, val_cutoff,
                                                test_cutoff, all_features,
                                                known_states=known_states,
                                                target_type=args.target_type,
                                                samp_pad=(14, nsteps_decode), trunc=trunc)
    else:
        trunc = 5 + epoch//args.trunc_patience
        nsteps_decode = random.randint(min_decode_times,
                                       min_decode_times + min(max_decode_times - min_decode_times, epoch//args.trunc_patience))

        train_val_dataset = AugmentedCovidDataset(train_val_df, unique_countries, val_cutoff, test_cutoff, all_features,
                                                  known_states=known_states, target_type=args.target_type,
                                                  nsteps_decode=nsteps_decode, trunc=trunc)

    train_val_loader = DataLoader(train_val_dataset, batch_size = args.batch_size, shuffle=True, num_workers = args.num_workers)

    train_epoch_stats = train_epoch(train_val_loader, net, optimizer, device, nsteps_decode,
                                    args.n_train_trajectories, args.elbo_type,
                                    args.reconstr_weight, criterion_func, args.noise_std,
                                    clip_value=args.clip_value, shift=args.shift)

    val_epoch_stats = predict(train_val_loader, net, device, n_elbo_samp=1,
                              elbo_type=args.elbo_type, noise_std=args.noise_std, shift=args.shift)

    print("Epoch: %d \t Train loss (last batch): %.3e \t Train Pred MSE: %.3e \t Train Reconstr MSE: %.3e\t Val MSE: %.3e "% (
           epoch, train_epoch_stats['train_loss'], train_epoch_stats['train_pred_mse'], train_epoch_stats['train_reconstr_mse'], val_epoch_stats['val_mse'],
    ), flush = True)

    if epoch % args.checkpoint_freq == 0:
        logs.append({
            'epoch': epoch,
            'train': train_epoch_stats,
            'val': val_epoch_stats
        })
        save_checkpoint(net, optimizer, scheduler,
                        epoch+1, es, torch.random.get_rng_state(), random.getstate(), logs)

    if epoch >= args.min_epochs:
        es(val_epoch_stats['val_mse'], es_path, net)
        scheduler.step(val_epoch_stats['val_mse'])

if args.data_type == 'debug' or not es_path.is_file():
    model_state_dict = net.state_dict()
else:
    model_state_dict = torch.load(es_path)

val_dataset = AugmentedCovidDataset(train_val_df, unique_countries, val_cutoff, test_cutoff,
                                    all_features, known_states=known_states,
                                    target_type=args.target_type, nsteps_decode=1, trunc=None)
val_loader = DataLoader(val_dataset, batch_size = args.batch_size * 2, shuffle = False, num_workers = args.num_workers)

predict_output = predict(val_loader, net, device, n_elbo_samp=50,
                         elbo_type=args.elbo_type, noise_std=args.noise_std)

pred_dfs, val_metrics = predictions_to_df(predict_output, train_val_df, val_cutoff,
                                          args.target_type)

pickle.dump(pred_dfs, (args.output_dir/'val_preds.pkl').open('wb'))
pickle.dump(logs, (args.output_dir/'logs.pkl').open('wb'))

with (args.output_dir/'val_metrics.json').open('w') as f:
    json.dump(val_metrics, f)
    print(json.dumps(val_metrics, indent = 4))

with (args.output_dir/'done').open('w') as f:
    f.write('done')
