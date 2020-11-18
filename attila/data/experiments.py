import ast
import json
import numpy as np

from attila.util.data import is_lst, is_numpy_array


def save_experiments(experiments, experiments_file):  # todo refactor (not here)
  out = [
    {
      k: v.tolist() if is_numpy_array(v) else v
      for k, v in experiment.items()
    }
    for experiment in experiments
  ]

  with open(experiments_file, 'w') as fp:
    json.dump(str(out), fp)


def load_experiments(experiments_file):  # todo refactor (not here)
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
