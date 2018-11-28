import keras
import keras.layers as kl


def discriminator(wd_rate=None):
    reg = keras.regularizers.l2(wd_rate) if wd_rate is not None else None
    inputs = kl.Input(shape=[200, 200, 3])
    x = kl.Conv2D(filters=30, kernel_size=[5, 5], use_bias=True, activation='relu', kernel_regularizer=reg)(inputs)
    x = kl.Conv2D(filters=30, kernel_size=[5, 5], use_bias=True, activation='relu', kernel_regularizer=reg)(x)
    x = kl.MaxPooling2D(pool_size=[2, 2])(x)
    x = kl.Conv2D(filters=60, kernel_size=[5, 5], use_bias=True, activation='relu', kernel_regularizer=reg)(x)
    x = kl.MaxPooling2D(pool_size=[2, 2])(x)
    x = kl.Conv2D(filters=60, kernel_size=[5, 5], use_bias=True, activation='relu', kernel_regularizer=reg)(x)
    x = kl.MaxPooling2D(pool_size=[2, 2])(x)
    x = kl.Conv2D(filters=60, kernel_size=[5, 5], use_bias=True, activation='relu', kernel_regularizer=reg)(x)
    x = kl.MaxPooling2D(pool_size=[2, 2])(x)
    x = kl.Conv2D(filters=60, kernel_size=[5, 5], use_bias=True, activation='relu', kernel_regularizer=reg)(x)
    x = kl.Flatten()(x)
    x = kl.Dense(units=100, activation='relu', use_bias=True)(x)
    x = kl.Dropout(rate=0.20)(x)
    x = kl.Dense(units=1, activation='sigmoid', use_bias=True)(x)
    return keras.Model(inputs=(inputs, ), outputs=(x, ))


def generator(wd_rate=None):
    reg = keras.regularizers.l2(wd_rate) if wd_rate is not None else None
    inputs = kl.Input(shape=[200, 200, 1])
    x = kl.Conv2D(filters=30, kernel_size=[5, 5], use_bias=True,
                  activation='relu', kernel_regularizer=reg, padding='same')(inputs)
    x = kl.Conv2D(filters=30, kernel_size=[5, 5], use_bias=True,
                  activation='relu', kernel_regularizer=reg, padding='same')(x)
    x = kl.Conv2D(filters=60, kernel_size=[5, 5], use_bias=True,
                  activation='relu', kernel_regularizer=reg, padding='same')(x)
    x = kl.Conv2D(filters=60, kernel_size=[5, 5], use_bias=True,
                  activation='relu', kernel_regularizer=reg, padding='same')(x)
    x = kl.Conv2D(filters=60, kernel_size=[5, 5], use_bias=True,
                  activation='relu', kernel_regularizer=reg, padding='same')(x)
    x = kl.Conv2D(filters=60, kernel_size=[5, 5], use_bias=True,
                  activation='relu', kernel_regularizer=reg, padding='same')(x)
    x = kl.Conv2D(filters=60, kernel_size=[5, 5], use_bias=True,
                  activation='relu', kernel_regularizer=reg, padding='same')(x)
    x = kl.Conv2D(filters=60, kernel_size=[5, 5], use_bias=True,
                  activation='relu', kernel_regularizer=reg, padding='same')(x)
    x = kl.Conv2D(filters=3, kernel_size=[5, 5], use_bias=True,
                  activation='relu', kernel_regularizer=reg, padding='same')(x)
    return keras.Model(inputs=(inputs, ), outputs=(x, ))


def merge_generator_discr(gen, discr):
    gen_new = generator()
    discr_new = discriminator()
    inputs = kl.Input(shape=[200, 200, 1])
    for i in range(len(gen.layers)):
        gen_new.layers[i].set_weights(gen.layers[i].get_weights())
    for i in range(len(discr.layers)):
        discr_new.layers[i].set_weights(discr.layers[i].get_weights())
        discr_new.layers[i].trainable = False
    x = gen_new(inputs)
    x = discr_new(x)
    return keras.Model(inputs=(inputs,), outputs=(x,))


def recreate_gen(net):
    gen_new = generator()
    for i in range(len(gen_new.layers)):
        gen_new.layers[i].set_weights(net.layers[1].layers[i].get_weights())
    return gen_new
