from parsers.img_loader import *

import keras
import keras.layers as kl

parser = DatasetParser()
training_set = TrainingSetIterator(parser=parser, shuffle=True, seed=5)

models = []


def thin_sequential_single(wd_rate=None):
    reg = keras.regularizers.l2(wd_rate)
    inputs = kl.Input(shape=[200, 200, 3])
    x = kl.Conv2D(filters=20, kernel_size=[5, 5], use_bias=True, activation='relu', kernel_regularizer=reg)(inputs)
    x = kl.MaxPooling2D(pool_size=[2, 2])(x)
    x = kl.Conv2D(filters=20, kernel_size=[5, 5], use_bias=True, activation='relu', kernel_regularizer=reg)(x)
    x = kl.MaxPooling2D(pool_size=[2, 2])(x)
    x = kl.Conv2D(filters=20, kernel_size=[5, 5], use_bias=True, activation='relu', kernel_regularizer=reg)(x)
    x = kl.MaxPooling2D(pool_size=[2, 2])(x)
    x = kl.Conv2D(filters=15, kernel_size=[5, 5], use_bias=True, activation='relu', kernel_regularizer=reg)(x)
    x = kl.MaxPooling2D(pool_size=[2, 2])(x)
    x = kl.Conv2D(filters=10, kernel_size=[5, 5], use_bias=True, activation='relu', kernel_regularizer=reg)(x)
    x = kl.Flatten()(x)
    x = kl.Dense(units=100, activation='relu', use_bias=True)(x)
    x = kl.Dropout(rate=0.20)(x)
    x = kl.Dense(units=29, activation='softmax', use_bias=True)(x)
    return keras.Model(inputs=(inputs, ), outputs=(x, ))


for c in classes_def:
    print('Building cnn for class ' + c)



