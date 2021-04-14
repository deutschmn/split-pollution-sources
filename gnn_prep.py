import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def transform_preds_back(time_npy, pred_npy, devices, data_start, melted=True):
    parsed_timestamps = pd.DataFrame(time_npy).applymap(pd.Timestamp.fromtimestamp)
    idx = np.where(parsed_timestamps[0] == data_start)[0][0] + 1 # there is an off-by-one error somewhere

    pred_index = pd.Index(parsed_timestamps[parsed_timestamps[0] == data_start].iloc[0], name='time')
    pred_df = pd.DataFrame(pred_npy[idx, :, :, 0], index=pred_index, columns=devices)

    if melted:
        return pred_df.melt(ignore_index=False, var_name='device_id', value_name='pm25')
    else:
        return pred_df

def transform_R_back(time_npy, R_npy, devices, data_start, prefix=""):
    if R_npy.shape[0] == 0:
        return None

    parsed_timestamps = pd.DataFrame(time_npy).applymap(pd.Timestamp.fromtimestamp)
    # TODO find out whether off-by-one is still there
    date_idx = np.where(parsed_timestamps[0] == data_start)[0][0] + 1 # there is an off-by-one error somewhere
    pred_index = pd.Index(parsed_timestamps[parsed_timestamps[0] == data_start].iloc[0], name='time')

    # if we use PM2.5 history, we have no R for the history
    # -> use only the last preds and don't look at the history
    pred_index = pred_index[-R_npy.shape[1]:]

    R_dict = {}
    for device_idx, device_id in enumerate(devices):
        R_dict[device_id] = pd.DataFrame(R_npy[date_idx, :, device_idx, :], index=pred_index, columns=devices)

    return R_dict