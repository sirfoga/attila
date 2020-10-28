def conv2d_block(n_filters, kernel_size=3, batchnorm=True):
    def _f(x):
        n_layers = 2

        for _ in range(n_layers):
            x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal", padding="same")(x)
            if batchnorm:
                x = BatchNormalization()(x)
            x = Activation("relu")(x)

        return x

    return _f


def contracting_block(n_filters, dropout, batchnorm, pooling=MaxPooling2D((2, 2))):
    def _f(x):
        x = conv2d_block(n_filters, kernel_size=3, batchnorm=batchnorm)(x)
        skip_conn = x  # save for up path

        x = pooling(x)
        x = Dropout(dropout)(x)
        return x, skip_conn

    return _f


def contracting_path(n_filters, n_layers, dropout, batchnorm):
    def _f(x):
        skip_conns = []
        current_n_filters = n_filters

        for _ in range(n_layers):
            x, s = contracting_block(current_n_filters, dropout, batchnorm)(x)

            current_n_filters *= 2
            skip_conns.append(s)

        return x, skip_conns

    return _f


def u_path(n_filters, batchnorm):
    def _f(x):
        x = conv2d_block(n_filters, kernel_size=3, batchnorm=batchnorm)(x)
        return x

    return _f


def expanding_block(n_filters, skip_conn, dropout, batchnorm):
    def _f(x):
        x = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding='same')(x)
        x = concatenate([x, skip_conn])
        x = Dropout(dropout)(x)
        x = conv2d_block(n_filters=n_filters, kernel_size=3, batchnorm=batchnorm)(x)

        return x

    return _f


def expanding_path(n_filters, skip_conns, dropout, batchnorm):
    def _f(x):
        current_n_filters = n_filters

        for skip_conn in reversed(skip_conns):
            x = expanding_block(current_n_filters, skip_conn, dropout, batchnorm)(x)

            current_n_filters /= 2

        return x

    return _f


def final_path(n_outs):
    def _f(x):
        x = Conv2D(n_outs, (1, 1), activation='sigmoid')(x)
        return x

    return _f


def get_unet(input_img, n_filters, n_layers, dropout=0.5, batchnorm=True):
    x, skip_conns = contracting_path(n_filters, n_layers, dropout, batchnorm)(input_img)
    x = u_path(n_filters * 2 ** n_layers, batchnorm)(x)
    x = expanding_path(n_filters * 2 ** n_layers, skip_conns, dropout, batchnorm)(x)
    outputs = final_path(1)(x)

    model = Model(inputs=[input_img], outputs=[outputs])
    return model

input_img = Input(input_shape)
model = get_unet(input_img, 64, 4, dropout=0.5, batchnorm=True)
# model.summary()
