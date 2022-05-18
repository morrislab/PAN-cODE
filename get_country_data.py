import pandas as pd
import numpy as np
import os
import sys
import requests
import io
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle
import json
from lib.Constants import *
import datetime

def download_csv(url, dropna = True):
    r = requests.get(url)
    if r.ok:
        data = r.content.decode('utf8')
        df = pd.read_csv(io.StringIO(data))
        if dropna:
            df = df.dropna(subset = ['key'], axis = 0, how = 'any')
        return df
    return None

INTER_URL = 'https://storage.googleapis.com/covid19-open-data/v2/oxford-government-response.csv'
TS_URL = 'https://storage.googleapis.com/covid19-open-data/v2/epidemiology.csv'
INDEX_URL = 'https://storage.googleapis.com/covid19-open-data/v2/index.csv'
HEALTH_URL = 'https://storage.googleapis.com/covid19-open-data/v2/health.csv'
WEATHER_URL = 'https://storage.googleapis.com/covid19-open-data/v2/weather.csv'
LOCATION_HUB_URL = 'https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master/data-locations/locations.csv'
OXFORD_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
ISO_COUNTRY_CODE_URL = 'https://raw.githubusercontent.com/GoogleCloudPlatform/covid-19-open-data/main/src/data/country_codes.csv'

inter = download_csv(INTER_URL)
ts = download_csv(TS_URL)
index = download_csv(INDEX_URL)
health = download_csv(HEALTH_URL)
weather = download_csv(WEATHER_URL)
oxford = download_csv(OXFORD_URL, dropna = False)[['CountryCode', 'RegionCode', 'Date', 'GovernmentResponseIndex', 'ContainmentHealthIndex', 'EconomicSupportIndex']]
iso_codes = download_csv(ISO_COUNTRY_CODE_URL, dropna = False)

index = index.loc[(index.aggregation_level == 0) | ((index.aggregation_level.isin([1, 2])) & (index.country_code == 'US')),
                 ['key', 'country_name', 'subregion1_name', 'subregion2_name', 'aggregation_level']]

a = set(ts.key.values)
b = set(inter.key.values)
keys_to_take = list(filter(lambda x: (x in a) and ((x in b) or (len(x.split('_'))) == 3 and x[:5] in b), index.key))

oxford = oxford.rename(columns={"CountryCode": "3166-1-alpha-3"}).merge(iso_codes).rename(columns={"3166-1-alpha-2": "country_code"})
oxford['key'] = oxford.apply(lambda x: x['RegionCode'] if not pd.isnull(x['RegionCode']) else x['key'], axis = 1)
oxford["date"] = oxford["Date"].apply(lambda x: datetime.datetime.strptime(str(x), "%Y%m%d").date().isoformat())

# for ind, row in index.iterrows():
#     if not pd.isnull(row['subregion1_name']):
#         index.loc[ind, 'country_name'] = row['country_name'] + '_' + row['subregion1_name']   
        
ts = ts[ts.key.isin(keys_to_take)].sort_values(by= ['key', 'date'], ascending = True)
inter = inter[inter.key.isin(keys_to_take)].sort_values(by= ['key', 'date'], ascending = True)
health = health[health.key.isin(keys_to_take)]
weather = weather[weather.key.isin(keys_to_take)].groupby(['date', 'key']).agg({'average_temperature': 'mean'}).reset_index()
oxford = oxford[oxford.key.isin(keys_to_take)].drop_duplicates(subset = ['date', 'key'])

inter = inter.merge(oxford[['key', 'date', 'GovernmentResponseIndex', 'ContainmentHealthIndex', 'EconomicSupportIndex']], on = ['key', 'date'], how = 'inner')

health_features = [i for i in list(health.columns) if i not in ['key']]
stringency_features = ['stringency_index', 'GovernmentResponseIndex', 'ContainmentHealthIndex', 'EconomicSupportIndex']
inter_features = ['school_closing', 'workplace_closing',
       'cancel_public_events', 'restrictions_on_gatherings',
       'public_transport_closing', 'stay_at_home_requirements',
       'restrictions_on_internal_movement', 'international_travel_controls',
        'income_support', 'debt_relief','public_information_campaigns',
       'testing_policy', 'contact_tracing',
          'facial_coverings', 'vaccination_policy']

inter = inter[['key', 'date'] + inter_features + stringency_features].fillna(method = 'ffill')

inter_features_dict_ohe = {'all_ts': [], 'condensed_ts': []}
inter_features_dict_ohe['basic_features'] = health_features
health.loc[health.key.str.startswith('US'), health_features] = health.loc[health.key.str.startswith('US'), health_features].fillna(value = health[health.key == 'US'].iloc[0], axis = 0)

new_rows_inter = []
for i in keys_to_take:
    if i.startswith('US_'):
        # duplicate health for each state and county
        if i not in health.key.values:
            row = health[health.key == 'US'].iloc[0]
            row['key'] = i
            health = health.append(row, ignore_index = True)

        # duplicate interventions for each county        
        if len(i.split('_')) == 3: # county level
            rows = inter[inter.key == i[:5]] # state level rows
            rows['key'] = i
            new_rows_inter.append(rows)

health[inter_features_dict_ohe['basic_features']] = (IterativeImputer(estimator = RandomForestRegressor(n_jobs = -1, random_state = 42))
                                                 .fit_transform(health[inter_features_dict_ohe['basic_features']]))
scaler = StandardScaler()
health[inter_features_dict_ohe['basic_features']] = scaler.fit_transform(health[inter_features_dict_ohe['basic_features']])

scalers = {
    'basic_features': scaler
}

inter = pd.concat([inter] + new_rows_inter, ignore_index = True)

for i in inter_features:
    min_val = int(inter[i].min())
    max_val = int(inter[i].max())
    for j in range(min_val+1, max_val+1):
        temp_feat_name = '%s==%s'%(i, j)
        inter[temp_feat_name] = (inter[i] == j).astype(int)
        inter_features_dict_ohe['all_ts'].append(temp_feat_name)

    temp_feat_name2 = '%s>0'%(i)
    inter[temp_feat_name2] = (inter[i] > 0).astype(int)
    inter_features_dict_ohe['condensed_ts'].append(temp_feat_name2)
    

ts = ts.rename(columns = {'total_confirmed': 'I', 'total_deceased': 'D',
                             'new_confirmed': 'delI', 
                             'new_deceased': 'delD'})
ts = ts[['key', 'date', 'I', 'D', 'delI', 'delD']]

df = (pd.merge(ts, inter, 
        on = ['key', 'date'], how = 'outer')
      .merge(health, on = ['key'], how = 'left')
      .merge(index, on = ['key'], how = 'left')
      .merge(weather, on = ['key', 'date'], how = 'left')
).sort_values(by = ['key', 'date'])

df['date'] = pd.to_datetime(df['date'])
for i in ['average_temperature'] + stringency_features + inter_features + inter_features_dict_ohe['condensed_ts'] + inter_features_dict_ohe['all_ts'] + inter_features_dict_ohe['basic_features']:
    df[i] = df[i].fillna(method = 'ffill')

np.random.seed(0)
df_orig = df[df['date'] <= max_date].set_index('key')

# select all days that occur after the country reaches min infected
df = []
for i in keys_to_take:
    temp = df_orig.loc[i]
    if (temp['I'] >= min_infected).sum() == 0: # never reached min infected
        continue
    first_row_id = (temp['I'] >= min_infected).values.argmax()
    to_add = temp.iloc[first_row_id:]
    df.append(to_add)

df = pd.concat(df)
keys_to_take = df.index.unique()

# add smoothed
for i in ['delI', 'delD']:
    df[i] = df[i].fillna(0)
    df.loc[df[i] < 0, i] = 0
    assert(pd.isnull(df[i]).sum() == 0)
    df[i + '_smoothed'] = 0
    for key in keys_to_take:
        df.loc[key, i + '_smoothed'] = df.loc[key, i].rolling(7).mean().fillna(0)
        
zero_time = df.groupby('key').first()['date'].rename('zero_time')
df = pd.merge(df, zero_time, on = 'key')
df['t'] = (df['date'] - df['zero_time']).dt.days
df = df[df.zero_time < (insufficient_data_cutoff - pd.Timedelta(days = min_data))].reset_index()

## Add stringency index and temperature as feature
for feat in ['average_temperature'] + stringency_features:
    scalers[feat] = StandardScaler()
    df[f'{feat}_norm'] = (scalers[feat].fit(df.loc[df.date < insufficient_data_cutoff, feat].values.reshape(-1, 1))
                                .transform(df[feat].values.reshape(-1, 1)))                              

    if feat in stringency_features:
        inter_features_dict_ohe['all_ts'].append(f'{feat}_norm')
        inter_features_dict_ohe['condensed_ts'].append(f'{feat}_norm')

inter_features_dict_ohe['weather_feature'] = ['average_temperature_norm']

keys_to_take = df.key.unique() 
id_mapping = {i:c for c, i in enumerate(keys_to_take)}
reverse_id_mapping = {c: i for c, i in enumerate(keys_to_take)}
df['country_id'] = df['key'].map(id_mapping)

covid_hub_mapping = download_csv(LOCATION_HUB_URL, dropna = False)[['abbreviation', 'location']].dropna().set_index('abbreviation')['location'].to_dict()
covid_hub_mapping = {'US_' + i : covid_hub_mapping[i] for i in covid_hub_mapping}
covid_hub_mapping['US'] = 'US'
df['covid_hub_id'] = df['key'].map(covid_hub_mapping) 

processed_data = {
    'countries': keys_to_take,
    'id_mapping': id_mapping,
    'reverse_id_mapping': reverse_id_mapping,
    'df': df,  
    'index': index
}

out_path = Path('./data/')
out_path.mkdir(exist_ok = True, parents = True)
df.to_pickle(out_path/'data.pkl')
json.dump(inter_features_dict_ohe, (out_path/'features.json').open('w'))
pickle.dump(scaler, (out_path/'scaler.pkl').open('wb'))
pickle.dump(processed_data, (out_path/'processed_data.pkl').open('wb'))

# sanity checks
for i in keys_to_take:
    temp = df[df.key == i]
    assert(temp.t.max()+1 == len(temp)), i
    assert(temp[['delI', 'delD']].isnull().sum().sum() == 0), i
    assert(temp[inter_features_dict_ohe['all_ts']].isnull().sum().sum() == 0), i