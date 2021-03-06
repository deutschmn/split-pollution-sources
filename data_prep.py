# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import geopandas as gpd

import datetime
import ndpretty

from tqdm.notebook import tqdm


# %%
ndpretty.default()

# %% [markdown]
# ## Load data
# 
# ### Metadata

# %%
def load_metadata():
    bj = pd.read_csv('data/bj_stations.csv')
    bj = bj.rename(columns={'deviceid': 'device_id'})
    bj["city"] = "Beijing"

    hebei = pd.read_csv('data/hebei_translated.csv')

    meta = pd.concat([bj, hebei])
    meta = gpd.GeoDataFrame(meta, geometry=gpd.points_from_xy(meta.lon, meta.lat))
    return meta

# load_metadata()

# %% [markdown]
# ### Air quality measurements and weather data
# 
# First we set a start and an end date for the data considered in this project.

# %%
def load_measurements(devices, start='2016-01-01 00:00:00', end='2021-01-01 00:00:00'):
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    date_idx = pd.date_range(start, end, freq='h')

    measurements = []
    no_weather_devices = []

    air_features = None
    weather_features = None

    for device_id in tqdm(devices):
        air_df = pd.read_csv('data/air/' + device_id + '.csv')
        air_df['time'] = pd.to_datetime(air_df["time"])
        air_df = air_df.set_index('time', drop=False).reindex(index=date_idx)
        air_df['time'] = air_df.index
        air_df['device_id'] = device_id

        try:
            weather_df = pd.read_csv('data/wea/' + device_id + '.csv')
            weather_df['time'] = pd.to_datetime(weather_df["datetime"])
            del weather_df['datetime']

            both_df = air_df.merge(weather_df, on='time', how='left')

            measurements.append(both_df)
        except Exception as e:
            measurements.append(air_df)
            no_weather_devices.append(device_id)
        
        if air_features is None:
            air_features = list(air_df.columns.drop(['time', 'device_id']))

        if weather_features is None:
            weather_features = list(weather_df.columns.drop(['time']))

    all_df = pd.concat(measurements)
    
    num_devices = len(all_df['device_id'].unique())
    assert not all_df['time'].isnull().any(), "Time values mustn't be NaN"
    assert all_df.shape[0] / num_devices == len(date_idx), "Don't have entries for all time steps for all devices"

    print(f"Loaded air quality data from {len(measurements)} devices. No weather data for {no_weather_devices}")

    return all_df, air_features, weather_features

# devices = load_metadata()['device_id'].unique()
# load_measurements(devices)


# %%
class Event:
    def __init__(self, name, start, end, color):
        self.name = name
        self.start = start
        self.end = end
        self.color = color
    
    def __repr__(self):
        date_fmt = '%Y/%m/%d'
        return f" Event '{self.name}' ({self.start.strftime(date_fmt)} - {self.end.strftime(date_fmt)})"

    def __str__(self):
        return self.name

def get_beijing_lockdowns():
    ld1 = Event("LD 1", pd.to_datetime('2020-01-23'), pd.to_datetime('2020-04-08'), 'green')
    ld2 = Event("LD 2 (light)", pd.to_datetime('2020-06-15'), pd.to_datetime('2020-09-01'), 'blue')

    return [ld1, ld2]

# get_beijing_lockdowns()


# %%
class CityData:
    def __init__(self, name, metadata, measurements, air_features, weather_features, events):
        self.name = name
        self.metadata = metadata
        self.measurements = measurements
        self.air_features = air_features
        self.weather_features = weather_features
        self.events = events

    @property
    def devices(self):
        return self.metadata['device_id'].unique()

    @property
    def measurements_joined(self):
        return self.measurements.merge(self.metadata, on='device_id')

    def __repr__(self):
        return f"CityData '{self.name}'"

def load_beijing_data():
    devices = load_metadata()['device_id'].unique()
    return CityData('Beijing', load_metadata(), *load_measurements(devices), get_beijing_lockdowns())

# beijing_data = load_beijing_data()


