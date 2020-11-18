import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


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


def do_inference(model, weights_file, data, batch_size, verbose):
  model.load_weights(weights_file)
  return model.predict(data, verbose=verbose, batch_size=batch_size)


def do_evaluation():
  pass
