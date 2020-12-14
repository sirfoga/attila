import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from attila.util.plots import extract_preds, plot_sample

from attila.nn.models.unet import calc_img_shapes, build as build_model
from attila.nn.core import do_training, do_evaluation
from attila.nn.metrics import mean_IoU, DSC, calc_accuracy
from attila.nn.losses import weighted_categorical_crossentropy

from attila.data.prepare import get_weights_file, get_model_output_folder, describe
from attila.data.transform import crop_center_transformation
from attila.data.augment import do_augment

from attila.util.config import is_verbose
from attila.util.io import stuff2pickle
from attila.util.ml import are_gpu_avail
from attila.nn.sanity import do_sanity_check, get_steps_per_epoch
from attila.util.data import dict2numpy


def get_default_args(config):
    model_args = dict(
        n_filters=config.getint('unet', 'n filters'),
        n_layers=config.getint('unet', 'n layers'),
        kernel_size=config.getint('unet', 'conv size'),
        pool_size=config.getint('unet', 'pool size'),
        n_classes=config.getint('image', 'n classes'),
        final_activation=config.get('unet', 'final activation'),
        dropout=config.getfloat('unet', 'dropout'),
        batchnorm=config.getboolean('unet', 'batchnorm'),
        conv_inner_layers=config.getint('unet', 'n conv inner layers'),
        filter_mult=config.getint('unet', 'filter mult'),
    )

    magic_learning_rate = 2e-5  # @karpathy suggested 3e-4
    loss_weights = np.ones(config.getint('image', 'n classes'))
    
    compile_args = dict(
        optimizer=Adam(learning_rate=magic_learning_rate),
        loss=weighted_categorical_crossentropy(loss_weights), 
        metrics=['accuracy', mean_IoU(), DSC()]
    )

    return model_args, compile_args


def get_experiment_args(experiment, config):
    model_args, compile_args = get_default_args(config)
    args = dict(
        **model_args,
        padding=experiment['padding'],
        use_skip_conn=experiment['use_skip_conn'],
        use_se_block=experiment['use_se_block']
    )

    return args, compile_args


def get_model(experiment, config):
    args, compile_args = get_experiment_args(experiment, config)
    return build_model(**args), compile_args


def do_experiment(experiment, data, split_seed, config, plot_ids, optimizer=None, do_sanity_checks=False):
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


    def _get_datagen(X, y=None, augment=False, phase='training'):
        inp_gen_args = dict(
            featurewise_center=False,  # NOT ~ batchnorm on first layer
            featurewise_std_normalization=False,
            samplewise_center=False,
            samplewise_std_normalization=False,
        )
        out_gen_args = dict(
            featurewise_center=False,
            featurewise_std_normalization=False,
            samplewise_center=False,
            samplewise_std_normalization=False,
        )

        if phase == 'training':
            flowing_args = dict(
                batch_size=config.getint('training', 'batch size'),
                seed=split_seed,
                shuffle=True  # re-order samples each epoch
            )

            if augment:
                inp_gen_args['horizontal_flip'] = True
                inp_gen_args['vertical_flip'] = True
                inp_gen_args['preprocessing_function'] = do_augment
                
                out_gen_args['horizontal_flip'] = True
                out_gen_args['vertical_flip'] = True


            # do the train/test split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=config.getfloat('experiments', 'val size'),
                random_state=split_seed
            )

            seed = np.random.randint(0, 42)
            fit_args = dict(
                augment=augment,
                seed=seed
            )

            # create the training data generator (already flowing)
            train_inp_gen = ImageDataGenerator(**inp_gen_args)
            train_inp_gen.fit(X_train, **fit_args)
            train_inp_gen = train_inp_gen.flow(
                X_train,
                **flowing_args  # no need for 'subset' since we're flowing all
            )

            train_out_gen = ImageDataGenerator(**out_gen_args)
            train_out_gen.fit(y_train, **fit_args)
            train_out_gen = train_out_gen.flow(
                y_train,
                **flowing_args
            )

            # create the validation data generator (already flowing)
            val_inp_gen = ImageDataGenerator(**inp_gen_args)
            val_inp_gen.fit(X_val, **fit_args)
            val_inp_gen = val_inp_gen.flow(
                X_val,
                **flowing_args
            )

            val_out_gen = ImageDataGenerator(**out_gen_args)
            val_out_gen.fit(y_val, **fit_args)
            val_out_gen = val_out_gen.flow(
                y_val,
                **flowing_args
            )

            return zip(train_inp_gen, train_out_gen), zip(val_inp_gen, val_out_gen)
        elif phase == 'evaluation':
            gen = ImageDataGenerator(**inp_gen_args)
            gen.fit(X)
            return gen.flow(
                X,
                shuffle=False,
                batch_size=1  # 1 img at a time
            )

        return None


    def _prepare_data(data):
        (X_train, X_test, y_train, y_test) = data
        (img_inp_shape, img_out_shape) = _get_shapes(X_train)
        return (
            crop_center_transformation(img_inp_shape)(X_train),
            crop_center_transformation(img_inp_shape)(X_test),
            crop_center_transformation(img_out_shape)(y_train),
            crop_center_transformation(img_out_shape)(y_test),
        )


    (X_train, X_test, y_train, y_test) = _prepare_data(data)
    model, compile_args = get_model(experiment, config)
    if optimizer:
        compile_args['optimizer'] = optimizer

    n_epochs = config.getint('training', 'epochs')
    callbacks = [
        EarlyStopping(
            monitor='loss',
            patience=int(n_epochs / 2),  # run at least half epochs
            verbose=True
        ),
        ReduceLROnPlateau(
            factor=1e-1, 
            patience=5,  # no time to waste
            min_lr=1e-5,
            verbose=is_verbose('experiments', config)
        ),
    ]

    if do_sanity_checks:
        model.summary()

        n_samples = 8
        do_sanity_check(
            [
                y_train[np.random.randint(len(y_train))]
                for _ in range(n_samples)
            ],  # random training samples
            [
                calc_accuracy(),
                mean_IoU(),
                DSC(),
                # todo loss
            ],
            config
        )


    if are_gpu_avail():  # prevent CPU melting
        bs = config.getint('training', 'batch size')
        augment = config.getboolean('data', 'aug')

        results = do_training(
            model,
            _get_datagen(
                X_train,
                y_train,
                augment=augment,
                phase='training'
            ),
            get_steps_per_epoch(X_train, bs),
            get_steps_per_epoch(X_test, bs),
            n_epochs,
            compile_args,
            callbacks
        )

        gen = _get_datagen(
            X_test,
            phase='evaluation'
        )
        stats, preds = do_evaluation(
            model,
            (gen, y_test)
        )

        preds = extract_preds(
            X_test,
            y_test,
            preds,
            plot_ids
        )

        return {
            'history': dict2numpy(results.history),
            'stats': dict2numpy(stats),
            'preds': [ np.float32(x) for x in preds ]  # maybe different size
        }

    return {}


def do_experiments(experiments, data, split_seed, config, out_folder, plot_ids):
    if is_verbose('experiments', config):
        print('ready to perform {} experiments'.format(len(experiments)))

    for i, experiment in enumerate(experiments):
        if is_verbose('experiments', config):
            print('=== experiment #{} / {}: {}'.format(i + 1, len(experiments), experiment['name']))

        summary = do_experiment(
            experiment,
            data,
            split_seed,
            config,
            plot_ids
        )
        summary['name'] = experiment['name']

        model_out_folder = get_model_output_folder(out_folder, experiment['name'])
        out_f = model_out_folder / config.get('experiments', 'output file')
        stuff2pickle(summary, out_f)


def do_batch_experiments(experiments, data, config, out_folder):
    nruns = config.getint('experiments', 'nruns')
    (X, y) = data  # unpack
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.getfloat('experiments', 'test size'),
        random_state=42  # reproducible results
    )
    if is_verbose('experiments', config):
        print('training data: X ~ {}, y ~ {}'.format(X_train.shape, y_train.shape))
        print('testing data: X ~ {}, y ~ {}'.format(X_test.shape, y_test.shape))

    num_plots = 4
    plot_ids = np.random.randint(len(X_test), size=num_plots)

    for nrun in range(nruns):
        folder = out_folder / 'run-{}'.format(nrun)
        folder.mkdir(parents=True, exist_ok=True)

        if is_verbose('experiments', config):
            print('ready to perform #{} / {} batch of experiments'.format(nrun + 1, nruns))

        do_experiments(
            experiments,
            (X_train, X_test, y_train, y_test),
            nrun,
            config,
            folder,
            plot_ids=plot_ids
        )
