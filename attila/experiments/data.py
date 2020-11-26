import ast
import json
import numpy as np

from attila.util.data import is_lst, is_numpy_array


def experiment2dict(experiment):
    return {
        k: v.tolist() if is_numpy_array(v) else v
        for k, v in experiment.items()
    }


def save_experiment(experiment, f_path):
    out = experiment2dict(experiment)

    with open(f_path, 'w') as fp:
        json.dump(str(out), fp)


def save_experiments(experiments, experiments_file):
    out = [
        experiment2dict(experiment)
        for experiment in experiments
    ]

    with open(experiments_file, 'w') as fp:
        json.dump(str(out), fp)


def load_experiments(experiments_file):
    with open(experiments_file, 'r') as fp:
        experiments = ast.literal_eval(json.load(fp))
        experiments = [
            {
                k: np.array(v) if is_lst(v) else v
                for k, v in experiment.items()
            }
            for experiment in experiments
        ]

        return experiments
