import numpy as np
from tensorflow.keras import backend as K

from attila.nn.metrics import mean_IoU, DSC


def describe_model(model):
    trainable_params = sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])
    total_params = model.count_params()
    non_trainable_params = total_params - trainable_params
    n_layers = len(model.layers)

    print('=== model')
    print('= # layers: {}'.format(n_layers))
    print('= # total params: {}'.format(total_params))
    print('= # trainable params: {}'.format(trainable_params))
    print('= # non-trainable params: {}'.format(non_trainable_params))


def do_training(model, datagen, steps_per_epoch, n_epochs, compile_args, callbacks):
    describe_model(model)  # todo only if verbose

    model.compile(**compile_args)
    return None
    # return model.fit(
    #     datagen,
    #     steps_per_epoch=steps_per_epoch
    #     epochs=n_epochs,
    #     callbacks=callbacks,
    #     # workers=1,  # todo fix taurus writing
    #     # use_multiprocessing=False,
    # )  # history


def do_inference(model, data, verbose):
    (datagen, X_test) = data
    gen = datagen.flow(
        X_test,
        shuffle=False,
        batch_size=1  # 1 img at a time
    )

    return model.predict_generator(
        gen,
        verbose=1 if verbose else 0
    )


def do_evaluation(model, data, verbose):
    (datagen, X_test, y_test) = data
    metrics = [
        {
            'name': 'attila_metrics_mean_IoU',
            'callback': mean_IoU()
        },
        {
            'name': 'attila_metrics_DSC',
            'callback': DSC()
        }
    ]

    preds = do_inference(
        model,
        (datagen, X_test),
        batch_size,
        verbose
    )

    stats = {
        metric['name']: []
        for metric in metrics
    }

    gen = datagen.flow(
        y_test,
        shuffle=False,
        batch_size=1  # 1 img at a time
    )

    # breakpoint

    for ix in range(len(X_test)):
        for metric in metrics:
            metric_f = metric['callback']
            metric_val = metric_f(
                K.cast(y_test[ix, ...], dtype='float32'),
                K.cast(preds[ix, ...], dtype='float32')
            ).numpy()
            stats[metric['name']].append(metric_val)

    if verbose:
        print('=== evaluation stats')
        print('= metrics on test set (size: {})'.format(len(X_test)))
        f_out = '= {:>10} ~ mean {:.3f} median {:.3f} std {:.3f}'

        for metric_name, metric_vals in stats.items():
            metric_out = f_out.format(
                metric_name,
                np.mean(metric_vals),
                np.median(metric_vals),
                np.std(metric_vals)
            )
            print(metric_out)

    return stats, preds
