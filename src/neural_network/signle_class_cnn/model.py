from keras.engine.saving import load_model

from data.dataset_manager import DatasetManager
from parsers.img_loader import *

import keras
import keras.layers as kl
import numpy as np

import operator

def thin_sequential_single(wd_rate=None):
    reg = keras.regularizers.l2(wd_rate)
    inputs = kl.Input(shape=[200, 200, 3])
    x = kl.Conv2D(filters=15, kernel_size=[5, 5], use_bias=True, activation='relu', kernel_regularizer=reg)(inputs)
    x = kl.MaxPooling2D(pool_size=[2, 2])(x)
    x = kl.Conv2D(filters=15, kernel_size=[5, 5], use_bias=True, activation='relu', kernel_regularizer=reg)(x)
    x = kl.MaxPooling2D(pool_size=[2, 2])(x)
    x = kl.Conv2D(filters=10, kernel_size=[5, 5], use_bias=True, activation='relu', kernel_regularizer=reg)(x)
    x = kl.MaxPooling2D(pool_size=[2, 2])(x)
    x = kl.Conv2D(filters=10, kernel_size=[5, 5], use_bias=True, activation='relu', kernel_regularizer=reg)(x)
    x = kl.MaxPooling2D(pool_size=[2, 2])(x)
    x = kl.Conv2D(filters=5, kernel_size=[5, 5], use_bias=True, activation='relu', kernel_regularizer=reg)(x)
    x = kl.Flatten()(x)
    x = kl.Dense(units=50, activation='relu', use_bias=True)(x)
    x = kl.Dropout(rate=0.20)(x)
    x = kl.Dense(units=1, activation='sigmoid', use_bias=True)(x)
    return keras.Model(inputs=(inputs, ), outputs=(x, ))


iterations = 5
epochs_per_iteration = 1
chunk_size = 5800
batch_size = 50
wd = 1e-9
lr = 0.5e-4
optimizer = 'sgd'
loss = 'binary_crossentropy'
metrics = 'binary_accuracy'
model_name_pre = 'cnn_'
model_name_post = '_vs_ALL.h5'

load = False


def convert_labels_to_single_class(class_, labels):
    return np.array([1 if classes_def[np.where(v == 1)[0][0]] == class_ else 0 for v in labels])


def predict(x):

    """
    Returns the predicted a list of predicted class names using majority voting.
    Each class corresponds to one entry in the x vector
    :return: list of class names
    """

    scores = np.zeros(shape=(len(classes_def), len(x)))

    for idx, c in enumerate(classes_def):

        model_name = model_name_pre + c + model_name_post
        print('Loading model', model_name, 'and making predictions..')
        model = load_model(model_name)

        scores[idx] = model.predict(x).reshape(len(x))

    out = []

    for predictions in scores.T:
        # Majority vote
        max_idx_for_sample = 0
        max_prob_for_sample = 0
        for i, prediction in enumerate(predictions):
            if prediction > max_prob_for_sample:
                max_idx_for_sample = i

        out.append(classes_def[max_idx_for_sample])

    return out


def train_models():

    # Use multi threaded reader and Matteo's dataset manager for getting test data
    parser = DatasetParser(verbose=False)
    test_x, test_y = DatasetManager().get_test()

    for c in classes_def:

        print('Building cnn model for class ' + c)

        model = thin_sequential_single(wd)

        model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
        model.summary()

        test_y_single = convert_labels_to_single_class(c, test_y)

        for epoch in range(iterations):

            print('*' * 200)
            print('Starting epoch', epoch)

            training_set = TrainingSetIterator(parser=parser, shuffle=True, seed=5, chunksize=chunk_size)

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

                y = np.array(y)

                model.fit(
                    x=image_batch,
                    y=y,
                    batch_size=batch_size,
                    epochs=epochs_per_iteration,
                    verbose=1,
                    class_weight={
                        1: 28,
                        0: 1
                    },
                    validation_data=(test_x, test_y_single)
                )

        print('Binary model for class %s trained successfully' % c)

        model_name = model_name_pre + c + model_name_post

        model.save(model_name)


# test_x, test_y = DatasetManager().get_test()


train_models()
# print(predict(test_x))
