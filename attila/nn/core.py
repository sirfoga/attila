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


def do_training(model, X_train, X_val, y_train, y_val, batch_size, n_epochs, compile_args, callbacks, verbose):
    if verbose:
        describe_model(model)

    model.compile(**compile_args)
    return model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        validation_batch_size=batch_size,
        epochs=n_epochs,
        callbacks=callbacks,
        validation_data=(X_val, y_val),
        workers=1,
        use_multiprocessing=False,
    )  # history


def do_inference(model, X, batch_size, verbose):
    return model.predict(X, verbose=verbose, batch_size=batch_size)


def do_evaluation(model, X_test, y_test, batch_size, verbose):
    metrics = [
        {
            'name': 'batch_metric-mean_IoU',
            'callback': mean_IoU
        },
        {
            'name': 'batch_metric-DSC',
            'callback': DSC
        }
    ]

    preds = do_inference(
        model,
        X_test,
        batch_size,
        verbose
    )

    stats = {
        metric['name']: []
        for metric in metrics
    }

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
