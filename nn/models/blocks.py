from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Dropout, multiply, Dense, GlobalAveragePooling2D


def conv2d_block(n_filters, kernel_shape, padding, use_se_block, dropout=0.0, batchnorm=True, inner_layers=2):  # todo inner_layers as arg
  activation = 'relu'

  def _f(x):
    for _ in range(inner_layers):
      x = Conv2D(n_filters, kernel_shape, padding=padding)(x)

      if batchnorm:
        x = BatchNormalization()(x)

      x = Activation(activation=activation)(x)

      if use_se_block:
        x = se_block()(x)

      if dropout > 0:
        x = Dropout(dropout)(x)

    return x

  return _f


def se_block(r=16.0):
  def squeeze(x):
    return GlobalAveragePooling2D()(x)

  def fc(n_filters, activation):
    def _f(x):
      return Dense(n_filters, activation=activation, use_bias=False)(x)

    return _f

  def excite(x, n_channels, r):
    # todo why ? se = Reshape(se_shape)(se)
    x = fc(n_channels // r, 'relu')(x)
    x = fc(n_channels, 'sigmoid')(x)
    return x

  def _f(x):
    n_channels = x.shape[-1]

    inp = x  # save for later
    x = squeeze(x)
    x = excite(x, n_channels, r)
    return multiply([inp, x])

  return _f