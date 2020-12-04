import numpy as np
from tensorflow.keras import backend as K

from attila.nn.metrics import mean_IoU, DSC


def do_training(model, datagen, steps_per_epoch, n_epochs, compile_args, callbacks):
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


def do_inference(model, data, verbose=False):
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


def do_evaluation(model, data):
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
        batch_size
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

    for y, p in zip(gen, preds):
        for metric in metrics:
            metric_f = metric['callback']
            metric_val = metric_f(
                K.cast(y, dtype='float32'),  # breakpoint check type
                K.cast(p, dtype='float32')
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
