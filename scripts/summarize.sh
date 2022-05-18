#!/bin/bash

output_root="/scratch/hdd001/home/haoran/CovidForecast"
slurm_pre="--partition t4v2,p100,t4v1,rtx6000 --gres gpu:1 --mem 30gb -c 4 --job-name summarize --output $(realpath ../logs)/summarize_%A.log"

set -e 
cd ../

for d in ${output_root}/* ; do
    sbatch ${slurm_pre} --wrap="python -u summarize.py --experiment_name $(basename ${d}) --data_dir $(realpath ./data) \
        --output_dir ${output_root} --n_trajectories 50  --covid_hub_dir /scratch/hdd001/home/haoran/CovidProjections/covid19-forecast-hub/data-processed \
        --states_sum_counties --us_sum_states"
done

