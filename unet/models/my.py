def unet(input_size, n_classes=2, dropout=None, batchnorm=False):
    def conv_layer(n_filters, size, n_times=1, **kwargs):
        """ Just alias for keras.layers.Conv2D """

        def _f(x):
            for _ in range(n_times):  # todo wrapper
                x = Conv2D(n_filters, (size, size), **kwargs)(x)

            return x

        return _f

    def pool_layer(pool_size=(2, 2)):
        """ Just alias for keras.layers.MaxPooling2D """

        return MaxPooling2D(pool_size=pool_size)

    def contracting_block(n_filters, dropout, batchnorm):
        def _f(x):
            x = conv_layer(n_filters, 3, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
            skip_conn = x  # save to concat in corresponding skip connection later

            if batchnorm:
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

            x = pool_layer()(x)
            if dropout:
                x = Dropout(dropout)(x)

            return x, skip_conn

        return _f

    def contracting_path(n_filters, n_layers):
        def _f(x):
            current_n_filters = n_filters
            skip_conns = []

            for _ in range(n_layers):
                x, skip_conn = contracting_block(current_n_filters, dropout, batchnorm)(x)

                skip_conns.append(skip_conn)
                current_n_filters = current_n_filters * 2

            return x, skip_conns, current_n_filters

        return _f

    def u_path(n_filters):
        def _f(x):
            x = conv_layer(n_filters, 3, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
            return x

        return _f

    def upconv_block(n_filters, size, up_sampling_size):
        def _f(x):
            x = UpSampling2D(size = up_sampling_size)(x)
            x = conv_layer(n_filters, size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
            return x

        return _f

    def expanding_block(n_filters, skip_conn, dropout):
        def _f(x):
            x = upconv_block(n_filters, 2, (2,2))(x)  # up-convolution
            x = concatenate([skip_conn, x], axis=3)  # concat in z axis (3rd dim)

            if dropout:
                x = Dropout(dropout)(x)

            x = conv_layer(n_filters, 3, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
            return x

        return _f

    def expanding_path(skip_conns, n_filters):
        def _f(x):
            current_n_filters = n_filters
            for skip_conn in reversed(skip_conns):
                x = expanding_block(current_n_filters, skip_conn, dropout)(x)
                current_n_filters = current_n_filters / 2
            return x

        return _f

    def final_path():
        def _f(x):
            x = conv_layer(n_classes - 1, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
            # x = Activation(activation='softmax')(x)
            return x

        return _f

    def _build(input_size, n_filters, n_layers):
        x = Input(input_size)
        inputs = x  # copy

        x, skip_conns, current_n_filters = contracting_path(n_filters, n_layers)(x)  # down ...
        x = u_path(current_n_filters)(x)  # ... middke ...
        x = expanding_path(skip_conns, current_n_filters)(x)  # ... up ...
        x = final_path()(x)  # ... end!

        return Model(inputs = inputs, outputs = x)

    return _build(input_size, 64, 4)


model = unet(input_shape)
# model.summary()
