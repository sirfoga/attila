import numpy as np
from keras import backend as K

from attila.nn.metrics import mean_IoU, DSC


def do_training(model, data_gen, training_steps_per_epoch, validation_steps_per_epoch, n_epochs, compile_args, callbacks):
    (train_gen, valid_gen) = data_gen
    model.compile(**compile_args)
    return model.fit(
        train_gen,
        validation_data=valid_gen,
        validation_steps=validation_steps_per_epoch,
        steps_per_epoch=training_steps_per_epoch,
        epochs=n_epochs,
        callbacks=callbacks,
        # workers=1,  # todo fix taurus writing
        # use_multiprocessing=False,
    )  # history


def do_inference(model, gen, steps, verbose=False):
    return model.predict(
        gen,
        steps=steps,
        verbose=1 if verbose else 0
    )


def do_evaluation(model, data):
    (gen, y_test) = data
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
        gen,
        steps=len(y_test)
    )

    stats = {
        metric['name']: []
        for metric in metrics
    }
    for y, p in zip(y_test, preds):
        for metric in metrics:
            metric_f = metric['callback']
            metric_val = metric_f(
                K.cast(y, dtype='float32'),
                K.cast(p, dtype='float32')
            )
            stats[metric['name']].append(metric_val)
    
    return stats, preds
