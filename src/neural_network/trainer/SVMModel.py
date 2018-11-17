from sklearn import svm, metrics
from sklearn.externals import joblib
from src.data.dataset_manager import DatasetManager
import numpy as np
import time


class SvmModel:

    def __init__(self, train_set_provider, path=None):
        """
        Constructor of the KerasModel object.
        :param train_set_provider: src.data.dataset_manager.DatasetManager object. This is the object that will be
                                   to read chunks of the dataset, using its get_batch_train_multithreaded method
        :param path: if the model that has to be trained already exists as a .h5 file, then this must be different
                     from None and must correspond to its path. If this is different from None, parameter model is
                     ignored
        """
        if path is not None:
            self.mod = joblib.load(path)
        else:
            self.mod = svm.SVC(gamma='scale', decision_function_shape='ovo')
        self.train_set_provider = train_set_provider

    def train(self, train_set_chunk_size=10000, reader_workers=8):

        start_time = time.time()

        dm = DatasetManager()
        dm.shuffle_train()

        x_temp, y_temp = dm.get_batch_train_multithreaded(size=train_set_chunk_size, workers=reader_workers)

        # converting images to gray-scale
        x = np.mean(x_temp, 3)
        x = x.reshape((x_temp.shape[0], x_temp.shape[1] * x_temp.shape[2]))

        y = np.argmax(y_temp, 1)

        self.mod.fit(x, y)

        print("--- %s minutes elapsed---" % ((time.time() - start_time)/60))

    def save(self, path):
        joblib.dump(self.mod, path)

    def predict(self, x):
        return self.mod.predict(x)
