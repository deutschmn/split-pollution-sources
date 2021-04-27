import numpy as np
from gnn_prep import Experiment
from functools import reduce
import gnn_prep

def ensemblify(experiments, name):
    template_exp = list(experiments.values())[0]
    ensemble_preds = template_exp.preds.copy()

    ensemble_preds['pm25'] = reduce(lambda x, y: x + y, map(lambda x: x.preds['pm25'], experiments.values())) / len(experiments)
    return Experiment(name, template_exp.start_date, template_exp.devices, ensemble_preds, None, template_exp.time_npy, template_exp.date_idx)

def deep_ensemblify(experiments, name):
    template_exp = list(experiments.values())[0]
    date_idx, parsed_timestamps = gnn_prep.get_date_idx(template_exp.time_npy, template_exp.start_date)

    R_s = np.stack(list(map(lambda exp: exp.R_npy[exp.date_idx], experiments.values())))
    h_s = np.stack(list(map(lambda exp: exp.transform_h_back(), experiments.values())))
    R_ens = np.average(R_s, axis=0)
    h_ens = np.average(h_s, axis=0)

    pred_npy = R_ens @ h_ens
    preds = gnn_prep.transform_preds_back(parsed_timestamps, pred_npy, template_exp.devices, template_exp.start_date)
    R_in, R_out = gnn_prep.transform_R_back(parsed_timestamps, R_ens, template_exp.devices, template_exp.start_date)
    preds['time'] = preds.index
    return Experiment(name, template_exp.start_date, template_exp.devices, preds, pred_npy, template_exp.time_npy, template_exp.date_idx, R_in, R_out, None)