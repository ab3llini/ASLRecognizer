from sklearn import svm, metrics
from sklearn.externals import joblib
from src.data.dataset_manager import DatasetManager
import numpy as np
import time
import src.plots.Plot as pl


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
            self.mod = svm.SVC(decision_function_shape='ovo')
        self.train_set_provider = train_set_provider

    def train(self, train_set_chunk_size=10000, reader_workers=8):

        start_time = time.time()

        dm = DatasetManager()
        dm.shuffle_train()

        x_temp, y_temp = dm.get_batch_train_multithreaded(size=train_set_chunk_size, workers=reader_workers)
        x_temp2, y_temp2 = dm.get_test()
        # converting images to gray-scale
        x = np.mean(x_temp, 3)
        x = x.reshape((x_temp.shape[0], x_temp.shape[1] * x_temp.shape[2]))

        y = np.argmax(y_temp, 1)

        x_test = np.mean(x_temp2, 3)
        x_test = x_test.reshape((x_temp2.shape[0], x_temp2.shape[1] * x_temp2.shape[2]))

        y_test = np.argmax(y_temp2, 1)
        print(x.shape, y.shape)
        self.mod.fit(x, y)
        accuracy = self.mod.score(x_test, y_test)
        time_taken = ((time.time() - start_time)/60)
        print("--- %s minutes elapsed---" % time_taken)
        return time_taken, accuracy

    def save(self, path):
        joblib.dump(self.mod, path)

    def predict(self, x):
        return self.mod.predict(x)


if __name__ == '__main__':
    ds = DatasetManager()
    samples = [10, 100, 1000, 2000]
    accuracy = []
    for num in samples:
        mod = SvmModel(ds)
        t, a = mod.train(train_set_chunk_size=num, reader_workers=10)
        accuracy.append(a)
    pl.line(samples, accuracy, l1="", title="#SAMPLES VS TEST ACCURACY")