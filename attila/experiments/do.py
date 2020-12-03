import random
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from attila.util.plots import extract_preds, plot_sample

from attila.nn.models.unet import calc_img_shapes, build as build_model
from attila.nn.core import do_training, do_evaluation
from attila.nn.metrics import mean_IoU, DSC

from attila.data.prepare import get_weights_file, get_model_output_folder, describe
from attila.data.transform import crop_center_transformation

from attila.util.config import is_verbose
from attila.util.io import stuff2pickle


def get_experiments(options):
    # `options` is like
    # {
    #     'use_skip_conn': [True, False],
    #     'padding': ['same', 'valid'],
    #     'use_se_block': [True, False]
    # }
    return []  # todo


def get_default_args(config):
    model_args = {  # todo use dict(
        'n_filters': config.getint('unet', 'n filters'),
        'n_layers': config.getint('unet', 'n layers'),
        'kernel_size': config.getint('unet', 'conv size'),
        'pool_size': config.getint('unet', 'pool size'),
        'n_classes': config.getint('image', 'n classes'),
        'final_activation': config.get('unet', 'final activation'),  # todo try softmax
        'dropout': config.getfloat('unet', 'dropout'),
        'batchnorm': config.getboolean('unet', 'batchnorm'),
        'conv_inner_layers': config.getint('unet', 'n conv inner layers'),
        'filter_mult': config.getint('unet', 'filter mult'),
    }

    compile_args = {  # todo use dict(
        'optimizer': config.get('training', 'optimizer'),
        'loss': config.get('training', 'loss'),
        'metrics': ['accuracy', mean_IoU(), DSC()]
    }

    return model_args, compile_args


def get_experiment_args(experiment, config):
    model_args, compile_args = get_default_args(config)
    args = {  # todo use dict(
        **model_args,
        'padding': experiment['padding'],
        'use_skip_conn': experiment['use_skip_conn'],
        'use_se_block': experiment['use_se_block']
    }

    return args, compile_args


def get_model(experiment, config):
    args, compile_args = get_experiment_args(experiment, config)
    return build_model(**args), compile_args


def do_experiment(experiment, data, split_seed, steps_per_epoch, config, out_folder, plot_ids):
    def _get_shapes(inp):
        img_inp_shape = inp.shape[1: 2 + 1]  # width, height of input images
        return calc_img_shapes(
            config.getint('unet', 'n layers'),
            config.getint('unet', 'n conv layers'),
            config.getint('unet', 'conv size'),
            config.getint('unet', 'pool size'),
            experiment['padding'],
            adjust=experiment['padding'] == 'valid'
        )(img_inp_shape)


    def _get_datagen(X, y=None, augment=False, flowing=True):
        (img_inp_shape, img_out_shape) = _get_shapes(X)

        gen_args = dict(  # todo try normalize ?
            featurewise_center=False,
            featurewise_std_normalization=False,
            samplewise_center=False,
            samplewise_std_normalization=False,
        )

        if augment:
            gen_args['horizontal_flip'] = True
            gen_args['vertical_flip'] = True
            # todo add more augmentations

        gen = ImageDataGenerator(**gen_args)

        if flowing:
            flowing_args = {  # todo use dict(
                'batch_size': config.getint('training', 'batch size'),
                'seed': split_seed,
                'shuffle': True  # re-order samples each epoch
            }

            inp_gen = gen.flow(
                crop_center_transformation(imp_inp_shape)(X),
                **flowing_args
            )
            out_gen = gen.flow(
                crop_center_transformation(img_out_shape)(y),
                **flowing_args
            )
            return zip(inp_gen, out_gen)


        return gen


    (X_train, X_test, y_train, y_test) = data
    model, compile_args = get_model(experiment, config)
    verbose = is_verbose('experiments', config)
    callbacks = [
        EarlyStopping(patience=10, verbose=verbose),
        ReduceLROnPlateau(
            factor=1e-1,
            patience=3,
            min_lr=1e-5,
            verbose=verbose
        )
    ]

    results = do_training(
        model,
        _get_datagen(
            X_train,
            y_train,
            augment=config.getboolean('data', 'aug')
        ),
        steps_per_epoch,
        config.getint('training', 'epochs'),
        compile_args,
        callbacks
    )

    datagen = _get_datagen(X_test, augment=False, flowing=False)
    stats, preds = do_evaluation(
        model,
        (datagen, X_test, y_test)
        is_verbose('experiments', config)
    )

    model_out_folder = get_model_output_folder(out_folder, experiment['name'])
    preds = extract_preds(
        X_test,
        y_test,
        preds,
        plot_ids
    )

    summary = {
        'history': results.history,
        'stats': stats,
        'preds': preds
    }
    out_f = model_out_folder / config.get('experiments', 'output file')
    stuff2pickle(summary, out_f)

    return summary


def do_experiments(experiments, data, split_seed, steps_per_epoch, config, out_folder, plot_ids):
    if is_verbose('experiments', config):
        print('ready to perform {} experiments'.format(len(experiments)))

    for i, experiment in enumerate(experiments):
        if is_verbose('experiments', config):
            print('=== experiment #{} / {}: {}'.format(i + 1, len(experiments), experiment['name']))

        summary = do_experiment(
            experiment,
            data,
            split_seed,
            steps_per_epoch,
            config,
            out_folder,
            plot_ids
        )
        # todo save summary, preds with `stuff2pickle`


def do_batch_experiments(experiments, data, config, out_folder):
    nruns = config.getint('experiments', 'nruns')
    (X, y) = data  # unpack
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.getfloat('experiments', 'test size')
    )
    num_plots = 6
    plot_ids = [
        random.randint(0, len(X_test) - 1)
        for _ in range(num_plots)
    ]
    if is_verbose('experiments', config):
        print('testing data: X ~ {}, y ~ {}'.format(X_test.shape, y_test.shape))

    val_size = config.getfloat('experiments', 'val size') / (1 - config.getfloat('experiments', 'val size'))

    for nrun in range(nruns):
        folder = out_folder / 'run-{}'.format(nrun)
        folder.mkdir(parents=True, exist_ok=True)

        if is_verbose('experiments', config):
            print('ready to perform #{} / {} batch of experiments'.format(nrun + 1, nruns))

        n_train_samples = len(X_train) * (1 - val_size)  # assuming no data augm
        steps_per_epoch = n_train_samples / config.getint('training', 'batch size')  # let floor, will use not-used images next runs (most probably)

        do_experiments(
            experiments,
            (X_train, X_test, y_train, y_test),
            nrun,  # todo choose better seed
            steps_per_epoch,
            config,
            folder,
            plot_ids=plot_ids
        )
