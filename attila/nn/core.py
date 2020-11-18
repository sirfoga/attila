import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from metrics import mean_IoU, DSC


def do_training(model, X_train, X_val, y_train, y_val, model_file, batch_size, n_epochs, compile_args, verbose):
  callbacks = [  # todo as arg
    EarlyStopping(patience=10, verbose=verbose),
    ReduceLROnPlateau(factor=1e-1, patience=3, min_lr=1e-5, verbose=verbose),
    ModelCheckpoint(model_file, monitor='loss', verbose=verbose, save_best_only=True, save_weights_only=True)
  ]

  if verbose:
    trainable_params = sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])
    total_params = model.count_params()
    non_trainable_params = total_params - trainable_params

    print('=== model')
    print('= # layers: {}'.format(len(model.layers)))
    print('= # total params: {}'.format(total_params))
    print('= # trainable params: {}'.format(trainable_params))
    print('= # non-trainable params: {}'.format(non_trainable_params))

  model.compile(**compile_args)
  return model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=n_epochs,
    callbacks=callbacks,
    validation_data=(X_val, y_val)
  )  # history


def do_inference(model, weights_file, X, batch_size, verbose):
  model.load_weights(weights_file)
  return model.predict(X, verbose=verbose, batch_size=batch_size)


def do_evaluation(model, weights_file, X_test, y_test, batch_size, verbose):
  metrics = [  # todo as arg
    {
      'name': 'mean IoU',
      'callback': mean_IoU
    },
    {
      'name': 'DSC',
      'callback': lambda y_true, y_pred: DSC(K.constant(y_true), K.constant(y_pred), axis=[1, 2])
    }
  ]

  preds = do_inference(
    model,
    weights_file,
    X_test,
    batch_size,
    verbose
  )

  stats = {
    metric: []
    for metric in metrics
  }

  for ix in range(len(X_test)):
    for metric in metrics:
      metric_f = metric['callback']
      metric_val = metric_f(y_test[ix], preds[ix]).numpy()
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
