#!/bin/bash

set -e
cd ../

slurm_pre="--partition t4v2,p100,t4v1,rtx6000 --gres gpu:1 --mem 8gb -c 4 --job-name lode --output /scratch/ssd001/home/ruiashi/projects/CovidForecast/logs/lode_%A.log"

output_root="/scratch/ssd001/home/ruiashi/projects/CovidForecast/output"

for data_type in "US"; do
    python sweep.py launch \
        --model "lode" \
        --output_root "${output_root}" \
        --slurm_pre "${slurm_pre}" \
        --command_launcher "slurm" \
        --data_dir "/scratch/ssd001/home/ruiashi/CovidForecast/data/" \
        --n_weeks_ahead 4 \
        --data_type ${data_type} \
    	--randomize_training \
	--concat_cond_ts
done
