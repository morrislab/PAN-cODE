import pandas as pd

max_date = pd.Timestamp('2021-05-31')
insufficient_data_cutoff = pd.Timestamp('2021-01-01')
min_data = 120 # at least x days of data by insufficient_data_cutoff
min_infected = 20 # start data for each country when it reaches this many infected
