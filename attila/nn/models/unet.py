from tensorflow.keras import Input, Model
from tensorflow.keras.layers import MaxPooling2D, Conv2D, UpSampling2D, concatenate, Cropping2D
from attila.nn.models.blocks import se_block, conv2d_block


def g(n):
    if n <= 1:
        return 2

    return g(n - 1) * 2 + 4  # very MAGIC formula, aka 2^(n-1) + 2^(n-2) -1


def calc_crop_size(layer, conv_layers, conv_size, padding):
    if padding == 'valid':
        conv_crop = conv_layers * (conv_size - 1)
        return int(conv_crop * g(layer))

    return 0


def calc_out_size(n_layers, conv_layers, conv_size, pool_size, padding):
    """ calculate output size in a U-Net """

    conv_crop = conv_layers * (conv_size - 1)

    def _sub_tup(t, y):
        return (x - y for x in t)

    def _f(x):
        crop_size = calc_crop_size(n_layers, conv_layers, conv_size, padding)

        if padding == 'valid':
            x = _sub_tup(x, conv_crop)  # adjust
            x = _sub_tup(x, conv_crop)  # first
            x = _sub_tup(x, crop_size)  # until last concatenation
            x = _sub_tup(x, conv_crop)  # final
            x = tuple(int(_x) for _x in x)

        return x

    return _f


def contracting_block(n_filters, kernel_shape, pool_shape, padding, use_se_block, dropout, batchnorm, conv_inner_layers):
    pooling = MaxPooling2D(pool_shape)

    def _f(x):
        x = conv2d_block(n_filters, kernel_shape, padding, use_se_block, dropout, batchnorm, inner_layers=conv_inner_layers)(x)
        skip_conn = x  # save for expanding path
        x = pooling(x)  # ready for next block
        return x, skip_conn

    return _f


def contracting_path(n_filters, n_layers, kernel_shape, pool_shape, use_skip_conn, padding, use_se_block, dropout, batchnorm, conv_inner_layers, filter_mult):
    def _f(x):
        skip_conns = []
        current_n_filters = n_filters

        if padding == 'valid':
            crop_size = 4
            crop_size = int(crop_size / 2)  # side by side
            x = Cropping2D(crop_size)(x)

        for _ in range(n_layers):
            x, s = contracting_block(current_n_filters, kernel_shape, pool_shape, padding, use_se_block, dropout, batchnorm, conv_inner_layers)(x)
            current_n_filters = int(current_n_filters * filter_mult)

            if not use_skip_conn:
                s = None  # not to be used

            skip_conns.append(s)

        return x, skip_conns

    return _f


def middle_block(kernel_shape, padding, dropout, batchnorm, conv_inner_layers, filter_mult):
    use_se_block = False

    def _f(x):
        n_filters = int(x.shape[-1] * filter_mult)
        x = conv2d_block(n_filters, kernel_shape, padding, use_se_block, dropout, batchnorm, inner_layers=conv_inner_layers)(x)
        return x

    return _f


def expanding_block(n_filters, skip_conn, kernel_shape, pool_shape, padding, use_se_block, dropout, batchnorm, conv_inner_layers):
    upsampling = UpSampling2D(pool_shape)

    def _f(x):
        if use_se_block:
            x = se_block()(x)

        x = upsampling(x)

        using_skip_conn = not (skip_conn is None)
        if using_skip_conn:
            x = concatenate([x, skip_conn])

        x = conv2d_block(n_filters, kernel_shape, padding, use_se_block, dropout, batchnorm, inner_layers=conv_inner_layers)(x)
        return x

    return _f


def expanding_path(n_filters, skip_conns, kernel_shape, pool_shape, padding, use_se_block, dropout, batchnorm, conv_inner_layers, filter_mult):
    def _f(x):
        current_n_filters = n_filters

        for i, skip_conn in enumerate(reversed(skip_conns)):
            using_skip_conn = not (skip_conn is None)
            if using_skip_conn:
                crop_size = calc_crop_size(i + 1, 2, kernel_shape[0], padding)
                crop_size = int(crop_size / 2)  # side by side
                skip_conn = Cropping2D(crop_size)(skip_conn)

            x = expanding_block(current_n_filters, skip_conn, kernel_shape, pool_shape, padding, use_se_block, dropout, batchnorm, conv_inner_layers)(x)
            current_n_filters = int(current_n_filters / filter_mult)

        return x

    return _f


def final_path(n_classes, activation, padding, use_se_block):
    def _f(x):
        if use_se_block:
            x = se_block()(x)

        x = Conv2D(n_classes, (1, 1), padding=padding, activation=activation)(x)
        return x

    return _f


def unet_block(n_filters, n_layers, kernel_shape, pool_shape, n_classes, final_activation, padding, use_skip_conn, use_se_block, dropout, batchnorm, conv_inner_layers, filter_mult):
    def _f(x):
        x, skip_conns = contracting_path(
            n_filters,
            n_layers,
            kernel_shape,
            pool_shape,
            use_skip_conn,
            padding,
            use_se_block,
            dropout,
            batchnorm,
            conv_inner_layers,
            filter_mult
        )(x)
        
        x = middle_block(
            kernel_shape,
            padding,
            dropout,
            batchnorm,
            conv_inner_layers,
            filter_mult
        )(x)

        if use_skip_conn:
            current_n_filters = skip_conns[-1].shape[-1]
        else:
            current_n_filters = n_filters * filter_mult ** (n_layers - 1)

        x = expanding_path(
            current_n_filters,
            skip_conns,
            kernel_shape,
            pool_shape,
            padding,
            use_se_block,
            dropout,
            batchnorm,
            conv_inner_layers,
            filter_mult
        )(x)

        x = final_path(n_classes, final_activation, padding, use_se_block)(x)
        return x

    return _f


def build(n_filters, n_layers, kernel_size, pool_size, n_classes, final_activation, padding='same', use_skip_conn=True, use_se_block=False, dropout=0.0, batchnorm=False, conv_inner_layers=2, filter_mult=2):
    n_dim = 2  # 2D images only
    kernel_shape = (kernel_size, ) * n_dim
    pool_shape = (pool_size, ) * n_dim

    # input channels (1): grayscale
    # output channels (3): background, foreground, borders

    n_channels = 1
    inp = Input((None, None, n_channels))
    out = unet_block(
        n_filters,
        n_layers,
        kernel_shape,
        pool_shape,
        n_classes,
        final_activation,
        padding,
        use_skip_conn,
        use_se_block,
        dropout,
        batchnorm,
        conv_inner_layers,
        filter_mult
    )(inp)

    model = Model(inputs=inp, outputs=out)
    return model
