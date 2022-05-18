## To replicate the experiments in the paper:

### Step 0: Environment and Prerequisites
Run the following commands to clone this repo and create the Conda environment:

```
git clone https://github.com/IanShi1996/PAN-cODE.git
cd PAN-cODE
conda env create -f environment.yml
conda activate covidforecast
```

### Step 1: Obtaining the Data
Update timestamps as appropriate in `lib/Constants.py`, then run:
```
python get_country_data.py
```

### Step 2: Model Training
Train the PAN-cODE model using train.py. Hyperparameters can be specified using command line arguments, as documented in train.py.

### Step 3: Model Evaluation
The summarize.py script can be used to generate forecasts for integration with the COVID-19 forecast hub.
