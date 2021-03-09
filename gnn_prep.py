import pandas as pd
import numpy as np

def transform_preds_back(time_npy, pred_npy, devices, data_start, melted=True):
    parsed_timestamps = pd.DataFrame(time_npy).applymap(pd.Timestamp.fromtimestamp)
    idx = np.where(parsed_timestamps[0] == data_start)[0][0] + 1 # there is an off-by-one error somewhere

    pred_index = pd.Index(parsed_timestamps[parsed_timestamps[0] == data_start].iloc[0], name='time')
    pred_df = pd.DataFrame(pred_npy[idx, :, :, 0], index=pred_index, columns=devices)

    if melted:
        return pred_df.melt(ignore_index=False, var_name='device_id', value_name='pm25')
    else:
        return pred_df