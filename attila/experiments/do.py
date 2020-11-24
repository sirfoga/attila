import numpy as np
from sklearn.model_selection import train_test_split

from attila.util.plots import plot_preds, plot_history

from attila.nn.models.unet import calc_out_size, build as build_model
from attila.nn.core import do_training, do_evaluation
from attila.nn.metrics import mean_IoU, DSC

from attila.data.prepare import train_validate_test_split, get_weights_file, get_model_output_folder, describe
from attila.data.trans import crop_center_transformation, apply_transformations

from attila.util.config import is_verbose


def get_default_args(config):
    conv_kernel_size = 3
    pool_size = 2

    model_args = {
        'img_depth': config.getint('image', 'depth'),
        'n_filters': config.getint('unet', 'n filters'),
        'n_layers': config.getint('unet', 'n layers'),
        'kernel_size': conv_kernel_size,
        'pool_size': pool_size,
        'n_classes': 1,    # the other is 1 - ... (because it's a probability distribution)
        'final_activation': config.get('unet', 'final activation'),
        'dropout': config.getfloat('unet', 'dropout'),
        'batchnorm': config.getboolean('unet', 'batchnorm')
    }

    compile_args = {
        'optimizer': config.get('training', 'optimizer'),
        'loss': config.get('training', 'loss'),
        'metrics': ['accuracy', mean_IoU, DSC]
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


def do_experiment(experiment, data, config, out_folder):
    def _fix_data_shape(img_out_shape):
        def _f(x):
            output_shape = (*img_out_shape, config.getint('image', 'depth'))

            transformations = [
                crop_center_transformation(output_shape),
            ]
            x = apply_transformations(x, transformations)    # reduce output size
            x = np.array(x)
            return x

        return _f


    def _prepare_data(data):
        (X_train, X_val, X_test, y_train, y_val, y_test) = data    # unpack

        img_shape = y_train.shape[1: 2 + 1]    # width, height of input images
        img_out_shape = calc_out_size(
            config.getint('unet', 'n layers'),
            2,
            3,
            2,
            experiment['padding']
        )(img_shape)

        y_train = _fix_data_shape(img_out_shape)(y_train)
        y_val = _fix_data_shape(img_out_shape)(y_val)
        y_test = _fix_data_shape(img_out_shape)(y_test)

        return (X_train, X_val, X_test, y_train, y_val, y_test)


    (X_train, X_val, X_test, y_train, y_val, y_test) = _prepare_data(data)

    if is_verbose('experiments', config):
        describe(X_train, X_val, X_test, y_train, y_val, y_test)

    model, compile_args = get_model(experiment, config)
    weights_file = str(get_weights_file(out_folder, experiment['name']))

    results = do_training(
        model,
        X_train,
        X_val,
        y_train,
        y_val,
        weights_file,
        config.getint('training', 'batch size'),
        config.getint('training', 'epochs'),
        compile_args,
        is_verbose('experiments', config)
    )

    stats, preds = do_evaluation(
        model,
        weights_file,
        X_test,
        y_test,
        config.getint('training', 'batch size'),
        is_verbose('experiments', config)
    )

    plot_preds(
        X_test,
        y_test,
        preds,
        cmap=config.get('image', 'cmap'),
        title='model: {}'.format(experiment['name']),
        out_folder=get_model_output_folder(out_folder, experiment['name'])
    )

    return results, stats


def do_experiments(experiments, data, config, out_folder):
    if is_verbose('experiments', config):
        print('ready to perform {} experiments'.format(len(experiments)))

    for i, experiment in enumerate(experiments):
        if is_verbose('experiments', config):
            print('=== experiment # {} / {}: {}'.format(i + 1, len(experiments), experiment['name']))

        results, eval_stats = do_experiment(experiment, data, config, out_folder)
        experiments[i]['history'] = results.history
        experiments[i]['eval'] = eval_stats

    last_epochs = int(config.getint('training', 'epochs') * 0.8)
    plot_history(experiments, last=last_epochs, out_folder=out_folder)

    return experiments


def do_batch_experiments(experiments, data, config, out_folder):
    nruns = config.getint('experiments', 'nruns')
    X, y = data  # unpack
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=config.getfloat('experiments', 'test size')
    )
    val_size = config.getfloat('experiments', 'val size') / (1 - config.getfloat('experiments', 'test size'))

    for nrun in range(nruns):
        if is_verbose('experiments', config):
            print('ready to perform #{} / {} batch of experiments'.format(nrun + 1, nruns))

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size
        )  # different experiment different random seed

        # todo data augmentation
        data = (X_train, X_val, X_test, y_train, y_val, y_test)
        do_experiments(
            experiments,
            data,
            config,
            out_folder / 'run-{}'.format(nrun)
        )
