from pathlib import Path
import numpy as np

from tensorflow.keras.optimizers import Adam, SGD

from attila.data.parse import parse_data, get_data
from attila.data.prepare import get_train_test_split
from attila.experiments.do import do_batch_experiments, do_experiment
from attila.util.config import get_env
from attila.util.io import load_json, stuff2pickle, get_summary, dirs
from attila.util.ml import are_gpu_avail
from attila.util.plots import extract_preds, plot_history

_here = Path('.').resolve()


def main():
    """ learning rate VS metrics """

    config, data_path, out_path, _ = get_env(_here)
    out_path.mkdir(parents=True, exist_ok=True)  # rm and mkdir if existing

    images_path = data_path / config.get('data', 'images')
    masks_path = data_path / config.get('data', 'masks')

    raw = get_data(images_path, masks_path)
    X, y = parse_data(
        raw,
        (config.getint('image', 'width'), config.getint('image', 'height'))
    )

    X_train, X_test, y_train, y_test = get_train_test_split(
        X, 
        y, 
        config.getfloat('experiments', 'test size'), 
        verbose=True
    )

    num_plots = 8
    plot_ids = np.random.randint(len(X_test), size=num_plots)

    experiment = {
        "use_skip_conn": True,
        "padding": "same",
        "use_se_block": False,
        "name": "with_same"
    }  # vanilla

    config.set('training', 'batch size', '4')
    config.set('training', 'epochs', '50')
    config.set('data', 'aug', 'False')

    if are_gpu_avail():  # prevent CPU melting
        def _do_it(out_name, learning_rate):
            optimizer = Adam(learning_rate=learning_rate)
            summary = do_experiment(
                experiment,
                (X_train, X_test, y_train, y_test),
                0,
                config,
                plot_ids,
                optimizer=optimizer,
                do_sanity_checks=False,
                callbacks=[]
            )

            out_folder = out_path / 'trials' / 'lost' / out_name
            out_folder.mkdir(parents=True, exist_ok=True)
            out_f = out_folder / config.get('experiments', 'output file')
            stuff2pickle(summary, out_f)

        learning_rates = np.logspace(-6, -2, 20)
        
        for learning_rate in learning_rates:
            exp_name = '{:.10f}'.format(learning_rate)
            print('doing "{}" experiment'.format(exp_name))

            _do_it(exp_name, learning_rate)

    if not are_gpu_avail():  # only with CPU
        for folder in dirs(out_path / 'trials' / 'lost'):
            summary = get_summary(folder, config)
            
            _, results = experiment2tex(summary)
            for key, vals in results.items():
                vals = vals['all']
                print('{} (based on {} samples): {:.3f} +- {:.3f}'.format(key, len(vals), vals.mean(), vals.std()))

        # todo plot


if __name__ == '__main__':
  main()
