import numpy as np
from tensorflow.keras import backend as K
from attila.experiments.do import get_model


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

    for experiment in experiments:
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


def create_tex_experiments(experiments, config):
    create_tex_table_configurations(experiments, config)
    print()

    create_tex_table_results(experiments)
