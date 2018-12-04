import keras as k

def single_cnn(penalty = None, x=None):
    reg = k.regularizers.l2(penalty)
    inputs = k.layers.Input(shape=[200, 200, 3])
    x = k.layers.Conv2D(filters=100, kernel_size=[7, 7], strides=2, padding='valid',activation='relu', use_bias = True, kernel_regularizer = reg)(inputs)
    x = k.layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2), padding='valid')(x)
    x = k.layers.Conv2D(filters=20, kernel_size=[1, 1], strides=(1, 1), padding='valid',activation='relu', use_bias=True, kernel_regularizer=reg)(x)
    x = k.layers.Conv2D(filters=20, kernel_size=[3, 3], strides=(1, 1), padding='valid',activation='relu', use_bias=True, kernel_regularizer=reg)(x)

    x = k.layers.Conv2D(filters=20, kernel_size=[3, 3], strides=(1, 1), padding='valid', activation='relu', use_bias=True, kernel_regularizer=reg)(x)

    x = k.layers.Conv2D(filters=20, kernel_size=[1, 1], strides=(1, 1), padding='valid', activation='relu', use_bias=True, kernel_regularizer=reg)(x)
    x = k.layers.AveragePooling2D(pool_size=[2, 2], strides=(2, 2), padding='valid')(x)
    x = k.layers.Conv2D(filters=30, kernel_size=[1, 1], strides=(1, 1), padding='valid',activation='relu', use_bias=True, kernel_regularizer=reg)(x)
    x = k.layers.AveragePooling2D(pool_size=[2,2] , strides=(2, 2), padding='valid')(x)
    #x = k.layers.Conv2D(filters=10, kernel_size=[3, 3], strides=(1, 1), padding='valid',activation='relu', use_bias=True, kernel_regularizer=reg)(x)
    # x= k.layers.MaxPooling2D(pool_size=[2, 2], strides=(2, 2), padding='valid')(x)
    #x = k.layers.Conv2D(filters=20, kernel_size=[4, 4], strides=(1, 1), padding='valid',activation='relu', use_bias=True, kernel_regularizer=reg)(x)
    x = k.layers.Flatten()(x)
    x = k.layers.Dense(units=30, activation='relu', use_bias=True)(x)
    x = k.layers.Dropout(rate=0.30)(x)
    x = k.layers.Dense(units=1, activation='sigmoid',use_bias=True)(x)

    return k.Model(inputs=(inputs,), outputs=(x,))

def cnn2(penalty = None, x=None):
    reg = k.regularizers.l2(penalty)
    inputs = k.layers.Input(shape=[200, 200, 3])
    x = k.layers.Conv2D(filters=32, kernel_size=[7, 7], strides=2, padding='valid',activation='relu', use_bias = True, kernel_regularizer = reg)(inputs)
    x = k.layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2), padding='valid')(x)
    x = k.layers.Conv2D(filters=64, kernel_size=[1, 1], strides=(1, 1), padding='valid',activation='relu', use_bias=True, kernel_regularizer=reg)(x)
    x = k.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=(1, 1), padding='valid',activation='relu', use_bias=True, kernel_regularizer=reg)(x)
    x = k.layers.AveragePooling2D(pool_size=[2, 2], strides=(2, 2), padding='valid')(x)
    x = k.layers.Conv2D(filters=32, kernel_size=[3, 3], strides=(1, 1), padding='valid', activation='relu', use_bias=True, kernel_regularizer=reg)(x)
    x = k.layers.MaxPooling2D(pool_size=[2, 2], strides=(2, 2), padding='valid')(x)
    x = k.layers.Conv2D(filters=10, kernel_size=[5, 5], strides=(1, 1), padding='valid', activation='relu', use_bias=True, kernel_regularizer=reg)(x)
    #x = k.layers.AveragePooling2D(pool_size=[2, 2], strides=(2, 2), padding='valid')(x)
    #x = k.layers.Conv2D(filters=30, kernel_size=[1, 1], strides=(1, 1), padding='valid',activation='relu', use_bias=True, kernel_regularizer=reg)(x)
    #x = k.layers.AveragePooling2D(pool_size=[2,2] , strides=(2, 2), padding='valid')(x)
    #x = k.layers.Conv2D(filters=10, kernel_size=[3, 3], strides=(1, 1), padding='valid',activation='relu', use_bias=True, kernel_regularizer=reg)(x)
    # x= k.layers.MaxPooling2D(pool_size=[2, 2], strides=(2, 2), padding='valid')(x)
    #x = k.layers.Conv2D(filters=20, kernel_size=[4, 4], strides=(1, 1), padding='valid',activation='relu', use_bias=True, kernel_regularizer=reg)(x)
    x = k.layers.Flatten()(x)
    x = k.layers.Dense(units=100, activation='relu', use_bias=True)(x)
    x = k.layers.Dropout(rate=0.20)(x)
    x = k.layers.Dense(units=1, activation='sigmoid',use_bias=True)(x)

    return k.Model(inputs=(inputs,), outputs=(x,))