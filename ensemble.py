from gnn_prep import Experiment
from functools import reduce

def ensemblify(experiments, name):
    template_exp = list(experiments.values())[0]
    ensemble_preds = template_exp.preds.copy()

    ensemble_preds['pm25'] = reduce(lambda x, y: x + y, map(lambda x: x.preds['pm25'], experiments.values())) / len(experiments)
    return Experiment(name, template_exp.start_date, template_exp.devices, ensemble_preds)