import numpy as np
from tensorflow.keras import backend as K

from attila.experiments.data import load_experiments
from attila.experiments.do import get_model
from attila.util.config import is_verbose


def create_tex_table_configurations(experiments, config):
    row_table_f = '{} & {} & {} & {} & {} & {} \\\\'

    print('creating .tex table for {} experiments configurations\n'.format(len(experiments)))

    for experiment in experiments:
        model, _ = get_model(experiment, config)
        
        trainable_params = sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])
        n_layers = len(model.layers)

        row_table = row_table_f.format(
            experiment['name'],
            '\\cmark{}' if experiment['use_skip_conn'] else '\\xmark{}',
            '\\cmark{}' if experiment['use_se_block'] else '\\xmark{}',
            experiment['padding'],
            n_layers,
            trainable_params
        )
        print(row_table)


def create_tex_table_results(experiments):
    row_table_f = '{} & {} & {} \\\\'
    metric_keys = ['batch_metric-mean_IoU', 'batch_metric-mean_DSC']

    print('creating .tex table for {} experiments results\n'.format(len(experiments)))

    for experiment in experiments:
        results = experiment['eval']
        for key in metric_keys:  # save for later processing
            experiment[key] = np.mean(results[key])

    best_values = {
        key: np.max([
            experiment[key] for experiment in experiments
        ])
        for key in metric_keys
    }

    out = {}

    for experiment in experiments:
        out[experiment['name']] = {
            key: experiment[key]
            for key in metric_keys
        }

        for key in metric_keys:
            if experiment[key] == best_values[key]:
                experiment[key] = '\\textbf{{{:.3f}}}'.format(experiment[key])
            else:
                delta = 100 - 100 * experiment[key] / best_values[key]
                experiment[key] = '{:.3f} (-{:.1f} \%)'.format(experiment[key], delta)

        row_table = row_table_f.format(
            experiment['name'],
            *(experiment[key] for key in metric_keys)
        )
        print(row_table)

    return out


def create_tex_table_runs_results(runs):
    row_table_f = '{} & {} & {} \\\\'
    metric_keys = list(list(runs[0].values())[0].keys())  # keys of all models
    model_names = list(runs[0].keys())

    print('creating .tex table for {} runs\n'.format(len(runs)))

    out = {
        model: {
            key: np.mean([
                run[model][key]
                for run in runs
            ])  # across all runs
            for key in metric_keys
        }
        for model in model_names
    }

    best_values = {
        key: np.max([
            out[model_name][key]
            for model_name in model_names
        ])  # across all models
        for key in metric_keys
    }

    for model_name, results in out.items():
        for key in metric_keys:
            if results[key] == best_values[key]:
                results[key] = '\\textbf{{{:.3f}}}'.format(results[key])
            else:
                delta = 100 - 100 * results[key] / best_values[key]
                results[key] = '{:.3f} (-{:.1f} \%)'.format(results[key], delta)

        row_table = row_table_f.format(
            model_name,
            *(results[key] for key in metric_keys)
        )
        print(row_table)

    return out


def create_tex_experiments(config, out_folder):
    nruns = config.getint('experiments', 'nruns')
    all_runs = []
    for nrun in range(nruns):
        folder = out_folder / 'run-{}'.format(nrun)
        results_file = folder / config.get('experiments', 'output file')
        results = load_experiments(results_file)

        if is_verbose('experiments', config):
            print('run #{}: loaded {} results from {}'.format(nrun + 1, len(results), results_file))

        create_tex_table_configurations(results, config)
        print()

        run_results = create_tex_table_results(results)
        print()

        all_runs.append(run_results)

    create_tex_table_runs_results(all_runs)
