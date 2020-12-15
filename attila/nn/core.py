import numpy as np
from keras import backend as K

from attila.nn.metrics import mean_IoU, DSC


def do_training(model, data_gen, compile_args, fit_args):
    (train_gen, valid_gen) = data_gen
    model.compile(**compile_args)
    
    return model.fit(
        train_gen,
        validation_data=valid_gen,
        **fit_args
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
        metric['name']: metric['callback'](
            K.cast(y_test, dtype='float32'),
            K.cast(preds, dtype='float32')
        )
        for metric in metrics
    }
    
    return stats, preds
