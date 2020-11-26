import random
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import train_test_split

from attila.util.plots import plot_preds, plot_history, plot_sample

from attila.nn.models.unet import calc_out_size, build as build_model
from attila.nn.core import do_training, do_evaluation
from attila.nn.metrics import mean_IoU, DSC

from attila.data.prepare import get_weights_file, get_model_output_folder, describe
from attila.data.transform import crop_center_transformation, do_transformations
from attila.data.augment import do_augmentations, flip, flop

from attila.util.config import is_verbose

from attila.experiments.data import save_experiments, save_experiment


def get_experiments(options):
    # `options` is like
    # {
    #     'use_skip_conn': [True, False],
    #     'padding': ['same', 'valid'],
    #     'use_se_block': [True, False]
    # }
    return []  # todo


def get_default_args(config):
    model_args = {
        'n_filters': config.getint('unet', 'n filters'),
        'n_layers': config.getint('unet', 'n layers'),
        'kernel_size': config.getint('unet', 'conv size'),
        'pool_size': config.getint('unet', 'pool size'),
        'n_classes': config.getint('image', 'n classes'),
        'final_activation': config.get('unet', 'final activation'),
        'dropout': config.getfloat('unet', 'dropout'),
        'batchnorm': config.getboolean('unet', 'batchnorm'),
        'conv_inner_layers': config.getint('unet', 'n conv inner layers'),
        'filter_mult': config.getint('unet', 'filter mult'),
    }

    compile_args = {
        'optimizer': config.get('training', 'optimizer'),
        'loss': config.get('training', 'loss'),
        'metrics': ['accuracy', mean_IoU(), DSC()]
    }

    return model_args, compile_args


def get_experiment_args(experiment, config):
    model_args, compile_args = get_default_args(config)
    args = {
        **model_args,
        'padding': experiment['padding'],
        'use_skip_conn': experiment['use_skip_conn'],
        'use_se_block': experiment['use_se_block']
    }

    return args, compile_args


def get_model(experiment, config):
    args, compile_args = get_experiment_args(experiment, config)
    return build_model(**args), compile_args


def do_experiment(experiment, data, config, out_folder, plot_ids):
    def _crop_data(img_out_shape):
        def _f(x):
            output_shape = (*img_out_shape, config.getint('image', 'n classes') - 1)
            return do_transformations(
                x,
                [
                    crop_center_transformation(output_shape),
                ]
            )

        return _f


    def _prepare_data(data):
        (X_train, X_val, X_test, y_train, y_val, y_test) = data  # unpack

        img_shape = y_train.shape[1: 2 + 1]  # width, height of input images
        img_out_shape = calc_out_size(
            config.getint('unet', 'n layers'),
            config.getint('unet', 'n conv layers'),
            config.getint('unet', 'conv size'),
            config.getint('unet', 'pool size'),
            experiment['padding']
        )(img_shape)

        y_train = _crop_data(img_out_shape)(y_train)
        y_val = _crop_data(img_out_shape)(y_val)
        y_test = _crop_data(img_out_shape)(y_test)

        return (X_train, X_val, X_test, y_train, y_val, y_test)


    (X_train, X_val, X_test, y_train, y_val, y_test) = _prepare_data(data)

    if is_verbose('experiments', config):
        describe(X_train, X_val, X_test, y_train, y_val, y_test)

    model, compile_args = get_model(experiment, config)
    verbose = is_verbose('experiments', config)
    weights_file = str(get_weights_file(out_folder, experiment['name']))
    callbacks = [
        EarlyStopping(patience=10, verbose=verbose),
        ReduceLROnPlateau(
            factor=1e-1,
            patience=3,
            min_lr=1e-5,
            verbose=verbose
        ),
        ModelCheckpoint(
            weights_file,
            monitor='loss',
            verbose=verbose,
            save_best_only=True,
            save_weights_only=True
            )
    ]

    results = do_training(
        model,
        X_train,
        X_val,
        y_train,
        y_val,
        config.getint('training', 'batch size'),
        config.getint('training', 'epochs'),
        compile_args,
        callbacks,
        verbose
    )

    model.load_weights(weights_file)  # loads best model
    stats, preds = do_evaluation(
        model,
        X_test,
        y_test,
        config.getint('training', 'batch size'),
        is_verbose('experiments', config)
    )

    model_out_folder = get_model_output_folder(out_folder, experiment['name'])
    plot_preds(
        X_test,
        y_test,
        preds,
        plot_ids,
        cmap=config.get('image', 'cmap'),
        title='model: {}'.format(experiment['name']),
        out_folder=model_out_folder
    )

    summary = {
        'history': results.history,
        'stats': stats
    }
    out_f = model_out_folder / 'summary.json'
    save_experiment(summary, out_f)

    last_epochs = int(config.getint('training', 'epochs') * 0.8)
    plot_history(
        summary['history'],
        last=last_epochs,
        out_folder=model_out_folder
    )

    return summary


def do_experiments(experiments, data, config, out_folder, plot_ids):
    if is_verbose('experiments', config):
        print('ready to perform {} experiments'.format(len(experiments)))

    for i, experiment in enumerate(experiments):
        if is_verbose('experiments', config):
            print('=== experiment #{} / {}: {}'.format(i + 1, len(experiments), experiment['name']))

        summary = do_experiment(experiment, data, config, out_folder, plot_ids)
        experiments[i]['history'] = summary['history']
        experiments[i]['stats'] = summary['stats']

    return experiments


def do_batch_experiments(experiments, data, config, out_folder):
    nruns = config.getint('experiments', 'nruns')
    (X, y) = data  # unpack
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=config.getfloat('experiments', 'test size')
    )
    num_plots = 6
    plot_ids = [
        random.randint(0, len(X_test) - 1)
        for _ in range(num_plots)
    ]
    if is_verbose('experiments', config):
        print('testing data: X ~ {}, y ~ {}'.format(X_test.shape, y_test.shape))

    val_size = config.getfloat('experiments', 'val size') / (1 - config.getfloat('experiments', 'test size'))

    for nrun in range(nruns):
        folder = out_folder / 'run-{}'.format(nrun)
        folder.mkdir(parents=True, exist_ok=True)

        if is_verbose('experiments', config):
            print('ready to perform #{} / {} batch of experiments'.format(nrun + 1, nruns))

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size
        )  # different experiment different random seed

        if config.getboolean('data', 'aug'):
            X_train, y_train = do_augmentations(
                X_train,
                y_train,
                [
                    flip(),
                    flop()
                ]
            )

            if is_verbose('experiments', config):
                print('augmented training data: X ~ {}, y ~ {}'.format(X_train.shape, y_train.shape))

        plot_sample(X_train, y_train, out_folder=folder)
        
        data = (X_train, X_val, X_test, y_train, y_val, y_test)
        results = do_experiments(
            experiments,
            data,
            config,
            folder,
            plot_ids=plot_ids
        )
        out_f = folder / config.get('experiments', 'output file')
        save_experiments(results, out_f)

        if is_verbose('experiments', config):
            print('done! output folder is {}'.format(folder))
