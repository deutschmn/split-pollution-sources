import pandas as pd
import numpy as np

class Experiment:
    def __init__(self, name, start_date, devices, preds, pred_npy, time_npy, date_idx, R_in=None, R_out=None, R_npy=None):
        self.name = name
        self.start_date = start_date
        self.devices = devices
        self.preds = preds
        self.pred_npy = pred_npy
        self.time_npy = time_npy
        self.date_idx = date_idx
        self.R_in = R_in
        self.R_out = R_out
        self.R_npy = R_npy

    @classmethod
    def load(cls, name, path, start_date, devices):
        time_npy = np.load(path + 'time.npy')
        pred_npy = np.load(path + 'predict.npy')

        date_idx, parsed_timestamps = get_date_idx(time_npy, start_date)
        preds = transform_preds_back(parsed_timestamps, pred_npy[date_idx], devices, start_date, melted=True)

        try:
            R_npy = np.load(path + 'R.npy')
            R_in, R_out = transform_R_back(parsed_timestamps, R_npy[date_idx], devices, start_date)
        except FileNotFoundError as e:
            print(f"Couldn't load R.npy for '{name}': {e}")
            R_in, R_out, R_npy = None, None, None

        preds['time'] = preds.index
        return cls(name, start_date, devices, preds, pred_npy, time_npy, date_idx, R_in, R_out, R_npy)

    def transform_h_back(self):
        c = self.pred_npy[self.date_idx]
        R = self.R_npy[self.date_idx]
        h = np.linalg.inv(R) @ c
        return h

    def __repr__(self):
        return f"Experiment '{self.name}'"


def get_date_idx(time_npy, data_start):
    parsed_timestamps = pd.DataFrame(time_npy).applymap(pd.Timestamp.fromtimestamp)
    # TODO find out whether off-by-one is still there
    date_idx = np.where(parsed_timestamps[0] == data_start)[0][0] + 1 # there is an off-by-one error somewhere
    return date_idx, parsed_timestamps


def transform_preds_back(parsed_timestamps, pred_npy, devices, data_start, melted=True):
    pred_index = pd.Index(parsed_timestamps[parsed_timestamps[0] == data_start].iloc[0], name='time')
    pred_df = pd.DataFrame(pred_npy[:, :, 0], index=pred_index, columns=devices)

    if melted:
        return pred_df.melt(ignore_index=False, var_name='device_id', value_name='pm25')
    else:
        return pred_df


def transform_R_back(parsed_timestamps, R_npy, devices, data_start):
    if R_npy.shape[0] == 0:
        return None, None

    pred_index = pd.Index(parsed_timestamps[parsed_timestamps[0] == data_start].iloc[0], name='time')

    # if we use PM2.5 history, we have no R for the history
    # -> use only the last preds and don't look at the history
    pred_index = pred_index[-R_npy.shape[0]:]

    R_in = {}
    R_out = {}
    for device_idx, device_id in enumerate(devices):
        R_in[device_id] = pd.DataFrame(R_npy[:, device_idx, :], index=pred_index, columns=devices)
        R_out[device_id] = pd.DataFrame(R_npy[:, :, device_idx], index=pred_index, columns=devices)

    return R_in, R_out

