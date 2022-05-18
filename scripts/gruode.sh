#!/bin/bash

set -e
cd ../

slurm_pre="--partition t4v2,p100,t4v1,rtx6000 --gres gpu:1 --mem 20gb -c 4 --job-name gruode --output /scratch/ssd001/home/haoran/projects/CovidForecast/logs/gruode_%A.log"

output_root="/scratch/hdd001/home/haoran/CovidForecast"

for anchor_date in '2021-03-08' '2020-12-28'; do
    for weeks in 4; do
        for randomize in "--randomize_training" ""; do
            for data_type in "US"; do
                python sweep.py launch \
                    --model "gruode" \
                    --output_root "${output_root}" \
                    --slurm_pre "${slurm_pre}" \
                    --command_launcher "slurm" \
                    --data_dir "/scratch/ssd001/home/haoran/projects/CovidForecast/data/" \
                    --n_weeks_ahead ${weeks} \
                    --concat_cond_ts \
                    --data_type ${data_type} \
                    --anchor_date ${anchor_date} \
                    ${randomize}
            done
        done
    done
done