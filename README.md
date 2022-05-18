## To replicate the experiments in the paper:

### Step 0: Environment and Prerequisites
Run the following commands to clone this repo and create the Conda environment:

```
git clone https://github.com/hzhang0/CovidForecast.git
cd CovidForecast
conda env create -f environment.yml
conda activate covidforecast
```

### Step 1: Obtaining the Data
Update timestamps as appropriate in `lib/Constants.py`, then run:
```
python get_country_data.py
```