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


def parse_experiment(experiment):
    return {
        k: np.array(v) if is_lst(v) else v
        for k, v in experiment.items()
    }


def parse_experiments(raw_json):
    experiments = ast.literal_eval(raw_json)
    return [
        parse_experiment(experiment)
        for experiment in experiments
    ]


def load_experiments(experiments_file):
    with open(experiments_file, 'r') as fp:
        return parse_experiments(json.load(fp))


def load_experiment(experiments_file):
    with open(experiments_file, 'r') as fp:
        experiment = ast.literal_eval(json.load(fp))
        return parse_experiment(experiment)
