from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

from tensorflow.keras.optimizers import Adam, SGD

from attila.data.parse import parse_data, get_data
from attila.experiments.do import do_batch_experiments, do_experiment
from attila.util.config import get_env
from attila.util.io import load_json, stuff2pickle, get_summary, dirs
from attila.util.ml import are_gpu_avail
from attila.util.plots import extract_preds

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
        "use_skip_conn": False,
        "padding": "same",
        "use_se_block": False,
        "name": "with_same"
    }  # vanilla U-Net

    config.set('training', 'epochs', '50')

    optimizers = [
        {
            'name': 'adam, lr = 1e-3',
            'f': Adam(learning_rate=1e-3)
        },
        {
            'name': 'adam, lr = 1e-4',
            'f': Adam(learning_rate=1e-4)
        },
        {
            'name': 'adam, lr = 1e-5',
            'f': Adam(learning_rate=1e-5)
        },
        {
            'name': 'sgd, U-Net paper',
            'f': SGD(momentum=0.99)
        }
    ]

    if are_gpu_avail():  # prevent CPU melting
        for optim in optimizers:
            summary = do_experiment(
                experiment,
                (X_train, X_test, y_train, y_test),
                0,
                config,
                plot_ids,
                optimizer=optim['f'],
                do_sanity_checks=False
            )
            
            out_folder = out_path / 'trials' / 'optimizers' / optim['name'] / config.get('experiments', 'output file')
            out_folder.mkdir(parents=True, exist_ok=True)
            stuff2pickle(summary, out_f)

    if not are_gpu_avail():  # only with CPU
        for folder in dirs(out_path / 'trials' / 'optimizers'):
            summary = get_summary(folder, config)
                plot_history(
                summary['history'],
                out_folder=folder
            )
            print('history img saved in {}'.format(model_folder))


if __name__ == '__main__':
  main()
