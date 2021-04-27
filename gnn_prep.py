import pandas as pd
import numpy as np

class Experiment:
    def __init__(self, name, path, start_date, devices):
        self.name = name
        self.path = path
        self.start_date = start_date
        self.devices = devices
        self.preds, self.R_in, self.R_out, self.R_npy = self.load_experiment()

    def load_experiment(self):
        time_npy = np.load(self.path + 'time.npy')
        pred_npy = np.load(self.path + 'predict.npy')

        preds = transform_preds_back(time_npy, pred_npy, self.devices, self.start_date, melted=True)

        try:
            R_npy = np.load(self.path + 'R.npy')
            R_in, R_out = transform_R_back(time_npy, R_npy, self.devices, self.start_date)
        except FileNotFoundError as e:
            print(f"Couldn't load R.npy for '{self.name}': {e}")
            R_in, R_out, R_npy = None, None, None

        preds['time'] = preds.index
        return preds, R_in, R_out, R_npy

    def __repr__(self):
        return f"Experiment '{self.name}'"
        

def transform_preds_back(time_npy, pred_npy, devices, data_start, melted=True):
    parsed_timestamps = pd.DataFrame(time_npy).applymap(pd.Timestamp.fromtimestamp)
    idx = np.where(parsed_timestamps[0] == data_start)[0][0] + 1 # there is an off-by-one error somewhere

    pred_index = pd.Index(parsed_timestamps[parsed_timestamps[0] == data_start].iloc[0], name='time')
    pred_df = pd.DataFrame(pred_npy[idx, :, :, 0], index=pred_index, columns=devices)

    if melted:
        return pred_df.melt(ignore_index=False, var_name='device_id', value_name='pm25')
    else:
        return pred_df

def transform_R_back(time_npy, R_npy, devices, data_start):
    if R_npy.shape[0] == 0:
        return None, None

    parsed_timestamps = pd.DataFrame(time_npy).applymap(pd.Timestamp.fromtimestamp)
    # TODO find out whether off-by-one is still there
    date_idx = np.where(parsed_timestamps[0] == data_start)[0][0] + 1 # there is an off-by-one error somewhere
    pred_index = pd.Index(parsed_timestamps[parsed_timestamps[0] == data_start].iloc[0], name='time')

    # if we use PM2.5 history, we have no R for the history
    # -> use only the last preds and don't look at the history
    pred_index = pred_index[-R_npy.shape[1]:]

    R_in = {}
    R_out = {}
    for device_idx, device_id in enumerate(devices):
        R_in[device_id] = pd.DataFrame(R_npy[date_idx, :, device_idx, :], index=pred_index, columns=devices)
        R_out[device_id] = pd.DataFrame(R_npy[date_idx, :, :, device_idx], index=pred_index, columns=devices)

    return R_in, R_out