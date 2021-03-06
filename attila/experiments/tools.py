import numpy as np
from keras import backend as K

from attila.experiments.do import get_model

from attila.util.io import append_rows2text, load_pickle


def runs2tex(runs, models_config, metric_keys=['attila_metrics_mean_IoU', 'attila_metrics_DSC']):
    models_runned = set()
    for run in runs:
        for key in run:
            models_runned.add(key)

    def _get_across_runs(model_names, metric_keys, runs):
        across_runs = {}  # model: { metric : { mean, std}}
        
        for model in models_runned:  # todo only the ones in UNION runs
            across_runs[model] = {}

            for key in metric_keys:
                _vals = np.ravel([
                    run[model][key]['all']
                    for run in runs
                    if model in run
                ])

                across_runs[model][key] = {
                    'mean': np.mean(_vals),
                    'std': np.std(_vals)
                }

        best_values = {  # key: max
            key: np.max([
                across_runs[model][key]['mean']
                for model in models_runned
            ])  # across all models
            for key in metric_keys
        }

        return across_runs, best_values


    print('creating .tex table for {} runs'.format(len(runs)))
    across_runs, best_values = _get_across_runs(
        models_runned,
        metric_keys,
        runs
    )

    row_table_f = '{} & {} & {} & {} & {} \\\\'
    #                  skip?    padding    DSC
    #             name       SE?      IoU

    rows = []
    epsilon = 5e-4
    for model, results in across_runs.items():
        _2tex = {}

        for key in metric_keys:
            if results[key]['mean'] >= best_values[key] - epsilon:
                _2tex[key] = '\\textbf{{{:.3f}}}'.format(results[key]['mean'])
            else:
                _2tex[key] = '{:.3f}'.format(results[key]['mean'])

            if results[key]['std'] >= 1e-4:  # there is a meaningful STD to show
                _2tex[key] += ' $\\pm$ {:.3f}'.format(results[key]['std'])

        experiment = [
            exp
            for exp in models_config
            if exp['name'] == model
        ][0]  # find experiment config

        row_table = row_table_f.format(
            '\\cmark{}' if experiment['use_skip_conn'] else '\\xmark{}',
            '\\cmark{}' if experiment['use_se_block'] else '\\xmark{}',
            '\\texttt{' + experiment['padding'] + '}',
            *(_2tex[key] for key in metric_keys)
        )
        rows.append(row_table)

    return rows, across_runs
