import pandas as pd
import numpy as np

class Experiment:
    def __init__(self, name, start_date, devices, preds, pred_npy, time_npy, date_idx, parsed_timestamps, R_in=None, R_out=None, R_npy=None):
        self.name = name
        self.start_date = start_date
        self.devices = devices
        self.preds = preds
        self.pred_npy = pred_npy
        self.time_npy = time_npy
        self.date_idx = date_idx
        self.parsed_timestamps = parsed_timestamps
        self.R_in = R_in
        self.R_out = R_out
        self.R_npy = R_npy

    @classmethod
    def load(cls, name, path, start_date, devices, default_column=-1):
        """Loads an experiment

        Args:
            name (str): name of the experiment
            path (str): where to load it from
            start_date (str or None): from which time stamp to look at prediction. if None, first prediction for each time step is returned
            devices (List[str]): list of devices

        Returns:
            Experiment
        """
        time_npy = np.load(path + 'time.npy')
        pred_npy = np.load(path + 'predict.npy')

        date_idx, parsed_timestamps = get_date_idx(time_npy, start_date)
        preds = transform_preds_back(date_idx, parsed_timestamps, pred_npy, devices, start_date, melted=True, default_column=default_column)

        try:
            R_npy = np.load(path + 'R.npy')
            if R_npy.shape[0] == 0:
                R_in, R_out = None, None
            else:
                R_in, R_out = transform_R_back(date_idx, parsed_timestamps, R_npy, devices, start_date, default_column)
        except FileNotFoundError as e:
            print(f"Couldn't load R.npy for '{name}': {e}")
            R_in, R_out, R_npy = None, None, None

        preds['time'] = preds.index
        return cls(name, start_date, devices, preds, pred_npy, time_npy, date_idx, parsed_timestamps, R_in, R_out, R_npy)

    def transform_h_back(self):
        c = self.pred_npy[self.date_idx]
        R = self.R_npy[self.date_idx]
        h = np.linalg.inv(R) @ c
        return h

    def __repr__(self):
        return f"Experiment '{self.name}'"


def get_date_idx(time_npy, data_start):
    parsed_timestamps = pd.DataFrame(time_npy).applymap(pd.Timestamp.fromtimestamp)
    if data_start is not None:
        # TODO find out whether off-by-one is still there
        date_idx = np.where(parsed_timestamps[0] == data_start)[0][0] + 1 # there is an off-by-one error somewhere
    else:
        date_idx = None
    return date_idx, parsed_timestamps


def transform_preds_back(date_idx, parsed_timestamps, pred_npy, devices, data_start, melted=True, default_column=-1):
    if data_start is None:
        # take the first prediction of each time step 
        # (only makes sense if hist_len = 0, otherwise we're probably plotting the ground truth)
        pred_index = pd.Index(parsed_timestamps[0])
        pred_df = pd.DataFrame(pred_npy[:,default_column,:,0], index=pred_index, columns=devices)
    else:
        pred_index = pd.Index(parsed_timestamps[parsed_timestamps[0] == data_start].iloc[0], name='time')
        pred_df = pd.DataFrame(pred_npy[date_idx, :, :, 0], index=pred_index, columns=devices)

    if melted:
        return pred_df.melt(ignore_index=False, var_name='device_id', value_name='pm25')
    else:
        return pred_df


def transform_R_back(date_idx, parsed_timestamps, R_npy, devices, data_start, default_column=-1):
    """Transforms the R values back

    Args:
        date_idx ([type]): [description]
        parsed_timestamps ([type]): [description]
        R_npy ([type]): [description]
        devices ([type]): [description]
        data_start ([type]): [description]
        default_column (int, optional): Which column to take if no data_start is provided. Defaults to -1.

    Returns:
        [type]: [description]
    """
    if data_start is None:
        pred_index = pd.Index(parsed_timestamps[0])
    else:
        pred_index = pd.Index(parsed_timestamps[parsed_timestamps[0] == data_start].iloc[0], name='time')

        # if we use PM2.5 history, we have no R for the history
        # -> use only the last preds and don't look at the history
        pred_index = pred_index[-R_npy.shape[1]:]

    R_in = {}
    R_out = {}
    for device_idx, device_id in enumerate(devices):
        if data_start is None:
            R_in[device_id] = pd.DataFrame(R_npy[:, default_column, device_idx, :], index=pred_index, columns=devices)
            R_out[device_id] = pd.DataFrame(R_npy[:, default_column, :, device_idx], index=pred_index, columns=devices)
        else:
            R_in[device_id] = pd.DataFrame(R_npy[date_idx, :, device_idx, :], index=pred_index, columns=devices)
            R_out[device_id] = pd.DataFrame(R_npy[date_idx, :, :, device_idx], index=pred_index, columns=devices)

    return R_in, R_out

