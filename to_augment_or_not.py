from pathlib import Path
import numpy as np

from tensorflow.keras.optimizers import Adam, SGD

from attila.data.parse import parse_data, get_data
from attila.data.prepare import get_train_test_split
from attila.experiments.do import do_batch_experiments, do_experiment
from attila.util.config import get_env
from attila.util.io import load_json, stuff2pickle, dirs, get_summary
from attila.util.plots import extract_preds, plot_history, plot_preds
from attila.util.ml import are_gpu_avail

_here = Path('.').resolve()


def main():
    """ # training images VS metrics """

    config, data_path, out_path, _ = get_env(_here)
    out_path.mkdir(parents=True, exist_ok=True)  # rm and mkdir if existing

    images_path = data_path / config.get('data', 'images')
    masks_path = data_path / config.get('data', 'masks')

    raw = get_data(images_path, masks_path)
    X, y = parse_data(
        raw,
        (config.getint('image', 'width'), config.getint('image', 'height'))
    )

    config.set('experiments', 'test size', '0.20')

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

    if are_gpu_avail():  # prevent CPU melting
        def _do_it(out_name, val_size, aug):
            config.set('experiments', 'val size', str(val_size))
            config.set('data', 'aug', str(aug))

            n_images = val_size * len(X_train)
            summary = do_experiment(
                experiment,
                (X_train, X_test, y_train, y_test),
                0,
                config,
                plot_ids,
                do_sanity_checks=False,
                callbacks=[]
            )
            summary['n images'] = n_images

            out_folder = out_path / 'trials' / 'to-aug-or-not' / out_name
            out_folder.mkdir(parents=True, exist_ok=True)
            out_f = out_folder / config.get('experiments', 'output file')
            stuff2pickle(summary, out_f)

        val_sizes = np.linspace(0.05, 0.95, 20)
        
        for val_size in val_sizes:
            exp_name = 'no-aug-{:.3f}'.format(val_size)
            print('doing "{}" experiment'.format(exp_name))

            _do_it(exp_name, val_size, False)

        for val_size in val_sizes:
            exp_name = 'aug-{:.3f}'.format(val_size)
            print('doing "{}" experiment'.format(exp_name))

            _do_it(exp_name, val_size, True)


if __name__ == '__main__':
  main()
