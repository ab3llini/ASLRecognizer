from data.dataset_manager import DatasetManager
from parsers.img_loader import *

import keras
import keras.layers as kl
import numpy as np


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
    x = kl.Dense(units=50, activation='relu', use_bias=True)(x)
    x = kl.Dropout(rate=0.20)(x)
    x = kl.Dense(units=1, activation='sigmoid', use_bias=True)(x)
    return keras.Model(inputs=(inputs, ), outputs=(x, ))


iterations = 20
epochs_per_iteration = 2
chunk_size = 3000
batch_size = 50
lr = 0.5e-2
optimizer = keras.optimizers.SGD(lr=lr)
loss = 'binary_crossentropy'
metrics = keras.metrics.binary_accuracy

# Use multi threaded reader and Matteo's dataset manager for getting test data
parser = DatasetParser(verbose=False)
training_set = TrainingSetIterator(parser=parser, shuffle=True, seed=5, chunksize=chunk_size)
test_x, test_y = DatasetManager().get_test()


for c in classes_def:

    print('Building cnn model for class ' + c)

    model = thin_sequential_single()

    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    model.summary()

    test_y = [1 if classes_def[np.where(v == 1)[0][0]] == c else 0 for v in test_y]

    for image_batch, label_batch in training_set:

        print('Loading new batch..')

        y = []
        for i, label in enumerate(label_batch):

            idx = np.where(label == 1)[0][0]
            class_for_image = classes_def[idx]
            if class_for_image == c:
                y.append(1)
            else:
                y.append(0)

        model.fit(
            x=image_batch,
            y=y,
            batch_size=batch_size,
            epochs=epochs_per_iteration,
            verbose=1,
            validation_data=(test_x, test_y)
        )

    print('Binary model for class %s trained successfully' % c)

    model_name = 'cnn_' + c + '_vs_ALL.h5'

    model.save(model_name)


