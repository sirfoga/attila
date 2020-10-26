def conv_layer(n_filters, size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'):
    """ Just alias for keras.layers.Conv2D """
    
    return Conv2D(n_filters, size, activation = activation, padding = padding, kernel_initializer = kernel_initializer)

def pool_layer(pool_size=(2, 2)):
    """ Just alias for keras.layers.MaxPooling2D """
    
    return MaxPooling2D(pool_size=pool_size)

def contracting_block(n_filters, x, dropout=0.4):
    x = conv_layer(n_filters, 3)(conv_layer(n_filters, 3)(x))
    skip_conn = x  # save to concat in corresponding skip connection later

    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = pool_layer()(x)
    x = Dropout(dropout)(x)

    return x, skip_conn

def contracting_path(x, start_layers=64, n_layers=4):
    current_n_layers = start_layers
    skip_conns = []
    
    for _ in range(n_layers):
        x, skip_conn = contracting_block(current_n_layers, x)
        
        skip_conns.append(skip_conn)
        current_n_layers *= 2
    
    return x, skip_conns

def upconv_block(x, n_filters, size, up_sampling_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'):
    x = UpSampling2D(size = up_sampling_size)(x)
    x = conv_layer(n_filters, size)(x)
    return x

def expanding_block(n_filters, x, skip_conn, dropout=0.4):
    x = upconv_block(x, n_filters, 2, (2,2))  # up-convolution
    x = concatenate([skip_conn, x], axis=3)  # concat in z axis (3rd dim)
    x = Dropout(dropout)(x)
    x = conv_layer(n_filters, 3)(conv_layer(n_filters, 3)(x))
    return x

def expanding_path(x, skip_conns, start_layers=512):
    current_n_layers = start_layers
    
    for skip_conn in reversed(skip_conns):
        x = expanding_block(current_n_layers, x, skip_conn)
        
        current_n_layers /= 2
    
    return x
    
def _unet(input_size = (128, 128, 1)):
    x = Input(input_size)
    inputs = x
    
    x, skip_conns = contracting_path(x)
    x = conv_layer(1024, 3)(conv_layer(1024, 3)(x))  # todo: dropout ?
    
    x = expanding_path(x, skip_conns)
    
    x = conv_layer(2, 3)(x)  # output segmentation map
    x = conv_layer(1, 1, activation = 'sigmoid')(x)
    
    outputs = x
    return inputs, outputs

def unet(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']):
    inputs, outputs = _unet()
    
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model
