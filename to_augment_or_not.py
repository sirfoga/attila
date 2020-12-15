from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

from tensorflow.keras.optimizers import Adam, SGD

from attila.data.parse import parse_data, get_data
from attila.experiments.do import do_batch_experiments, do_experiment
from attila.util.config import get_env
from attila.util.io import load_json, stuff2pickle, dirs, get_summary
from attila.util.plots import extract_preds, plot_history, plot_preds
from attila.experiments.tools import experiment2tex
from attila.util.ml import are_gpu_avail

_here = Path('.').resolve()


def main():
    config, data_path, out_path, _ = get_env(_here)
    out_path.mkdir(parents=True, exist_ok=True)  # rm and mkdir if existing

    images_path = data_path / config.get('data', 'images')
    masks_path = data_path / config.get('data', 'masks')

    raw = get_data(images_path, masks_path)
    X, y = parse_data(
        raw,
        (config.getint('image', 'width'), config.getint('image', 'height'))
    )

    config.set('experiments', 'test size', '0.95')  # or any other big amount (< 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.getfloat('experiments', 'test size'),
        random_state=42  # reproducible results
    )
    print('train/val data: X ~ {}, y ~ {}'.format(X_train.shape, y_train.shape))
    print('test data: X ~ {}, y ~ {}'.format(X_test.shape, y_test.shape))

    num_plots = 4
    plot_ids = np.random.randint(len(X_test), size=num_plots)

    experiment = {
        "use_skip_conn": True,
        "padding": "same",
        "use_se_block": False,
        "name": "with_same"
    }

    config.set('experiments', 'val size', '0.3')
    config.set('training', 'batch size', '4')
    config.set('training', 'epochs', '50')

    if are_gpu_avail():  # prevent CPU melting
        def _do_it(out_name):
            summary = do_experiment(experiment, (X_train, X_test, y_train, y_test), 0, config, plot_ids)

            out_folder = out_path / 'trials' / 'to-aug-or-not' / out_name
            out_folder.mkdir(parents=True, exist_ok=True)
            out_f = out_folder / config.get('experiments', 'output file')
            stuff2pickle(summary, out_f)


        config.set('data', 'aug', 'False')
        _do_it('no-aug')

        config.set('data', 'aug', 'True')
        _do_it('aug')

    if not are_gpu_avail():  # only with CPU
        for folder in dirs(out_path / 'trials' / 'to-aug-or-not'):
            summary = get_summary(folder, config)
            
            _, results = experiment2tex(summary)
            for key, vals in results.items():
                vals = vals['all']
                print('{} (based on {} samples): {:.3f} +- {:.3f}'.format(key, len(vals), vals.mean(), vals.std()))

            plot_history(
                summary['history'],
                last=0,
                out_folder=folder,
                loss_scale=[0, 0.5],
                met_scale=[0.5, 1.0]
            )
            print('history img saved in {}'.format(folder))


if __name__ == '__main__':
  main()
