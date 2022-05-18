#!/bin/bash
set -e
for i in "" "--use_infections"; do
    python evaluate_models.py 2021-03-08 2021-04-03 --forecast_hub_dir /scratch/hdd001/home/haoran/CovidProjections/covid19-forecast-hub --out_dir evaluations/ ${i}
    python evaluate_models.py 2021-03-08 2021-04-17 --forecast_hub_dir /scratch/hdd001/home/haoran/CovidProjections/covid19-forecast-hub --out_dir evaluations/ ${i}
    python evaluate_models.py 2020-12-28 2021-01-23 --forecast_hub_dir /scratch/hdd001/home/haoran/CovidProjections/covid19-forecast-hub --out_dir evaluations/ ${i}
    python evaluate_models.py 2020-12-28 2021-02-06 --forecast_hub_dir /scratch/hdd001/home/haoran/CovidProjections/covid19-forecast-hub --out_dir evaluations/ ${i}
done