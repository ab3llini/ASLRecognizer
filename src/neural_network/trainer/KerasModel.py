import keras as k
import time
import numpy as np
from numba import jit
import tqdm
import skimage.transform as skt
import src.data.utilities as u


class Kerasmodel:
    """
    In tasks like the one treated in this project the major issue is the dataset I/O.
    Since the dataset is too large to entirely fit in memory, it has to be read in
    chunks of a feasible size and the model must me trained on the chunks, one chunk per time.
    This class serves to abstract the task of reading the chunks and train on every chunk.
    Using an object of this class, all those operations are masked and the problem becomes
    transparent to the user of KerasModel.
    """
    def __init__(self, train_set_provider, opt, loss='categorical_crossentropy', metrics=k.metrics.categorical_accuracy,
                 path=None, model=None):
        """
        Constructor of the KerasModel object.
        :param train_set_provider: src.data.dataset_manager.DatasetManager object. This is the object that will be
                                   to read chunks of the dataset, using its get_batch_train_multithreaded method
        :param opt: optimizer that has to be used to train the model. Generally the Adam optimizer works well
        :param loss: the loss function to be minimized during the training. Default value is categorical_crossentropy, which
                     generally works fine in classification tasks
        :param metrics: metrics for which the model will be evaluated during training. Default value is accuracy.
        :param path: if the model that has to be trained already exists as a .h5 file, then this must be different
                     from None and must correspond to its path. If this is different from None, parameter model is
                     ignored
        :param model: if the keras model object that has to be trained has already been read or just created, then
                      it has to be given in this parameter. This is considered only if path is None.
        """
        if path is not None:
            self.mod = k.models.load_model(path)
        else:
            self.mod = model
        self.train_set_provider = train_set_provider
        self.mod.compile(optimizer=opt, loss=loss, metrics=[metrics])
        self.mod.summary()

    def train(self, iterations, epochs_per_iteration, train_batch_size=20, class_weight=None, validation=None,
              sleep_time=None, rest_every=None, train_set_chunk_size=10000, reader_workers=8, augment=False):
        """

        :param iterations: number of times a chunk is read from the dataset to train the model. Note that if
                           iterations < datasetsize/train_set_chunk_size then not the entire dataset is used for
                           for training and if iterations > datasetsize/train_set_chunk_size then the entire dataset
                           (or at least part of it) is used multiple times (this choice is generally positive)
        :param epochs_per_iteration: number of training epochs to run on each chunk.
                                     Ideal values are epochs_per_iteration=1 and iterations=ANY_BIG_NUMBER, so
                                     that this training procedure becomes equivalent to the one without the memory
                                     (big dataset) issue. Note that the training in the ideal case will be slower than
                                     in cases with epochs_per_iteration>1 and a smaller iterations parameter.
        :param train_batch_size: the keras model training batch size. Default value is 20.
        :param class_weight: use this parameter to give different weights to samples of different classes so that
                             they contribute differently to the error measure. Default is None (all equal)
        :param validation: a tuple (x_validation, y_validation) that is used during the training procedure to give
                           an evaluation of the model.
        :param sleep_time: amount of seconds the training procedure will remain paused every "rest_every" iteration.
                           Useful if the training is very long to avoid overheating, especially if GPU is being used.
                           This value is considered only if "rest_every" is a number.
        :param rest_every: Number of iterations after which the training procedure will be paused for "sleep_time"
                           seconds. Useful if the training is very long to avoid overheating, especially if GPU is
                           being used. By default, the training never stops. If this is a number, then "sleep_time"
                           MUST be a number as well.
        :param train_set_chunk_size: number of training samples to be read from memory in every iterations. Default
                                     value is 10000.
        :param reader_workers: number of threads to be used to speed up the dataset chunk reading procedure.
                               Default value is 8 threads. Set to 1 to have single thread.
        :return: nothing
        """

        for i in range(iterations):
            print("############# TRAIN ITERATION ", i)
            x, y = self.train_set_provider.get_batch_train_multithreaded(train_set_chunk_size, reader_workers)
            if augment:
                x, y = self.__augment(x, y, 1)
            self.mod.fit(x=x, y=y, batch_size=train_batch_size, epochs=epochs_per_iteration, verbose=1,
                         validation_data=validation, class_weight=class_weight)

            if rest_every is not None and (i+1) % rest_every == 0:
                print("############# TIME TO REST A BIT... #############")
                time.sleep(sleep_time)

    def __augment(self, x, y, n):
        samples = len(x)
        for i in tqdm.tqdm(range(samples)):
            newx, newy = self.__aug_shift(x[i], y[i], n)
            x = np.concatenate((x, newx))
            y = np.concatenate((y, newy))
        return x, y

    @jit
    def __aug_shift(self, x, y, n):
        ris = []
        risy = []
        shape = x.shape
        for _ in range(n):
            newx = np.zeros(shape, dtype=np.int16)

            lumin_change = 0.2 + np.random.rand() * 1.5
            pixels = np.random.randint(1, 6)
            dir = np.random.randint(0, 4)
            if dir == 0:  # left
                newx[:, :shape[1] - pixels] = x[:, pixels:shape[1]]
                newx[:, shape[1] - pixels:] = x[:, 0:pixels]
            if dir == 1:  # right
                newx[:, pixels:shape[1]] = x[:, :shape[1] - pixels]
                newx[:, 0:pixels] = x[:, shape[1] - pixels:]
            if dir == 2:  # up
                newx[:shape[0] - pixels, :] = x[pixels:shape[0], :]
                newx[shape[0] - pixels:, :] = x[:pixels, :]
            if dir == 3:  # down
                newx[pixels:shape[0], :] = x[:shape[0] - pixels, :]
                newx[:pixels, :] = x[shape[0] - pixels:, :]
            newx = newx * lumin_change
            newx[newx > 255] = 255
            newx = np.array(np.floor(newx), dtype=np.uint8)
            ris.append(newx)
            risy.append(y)
        return np.array(ris), np.array(risy)

    def save(self, path):
        self.mod.save(path)

    def predict(self, x):
        return self.mod.predict(x).squeeze()
