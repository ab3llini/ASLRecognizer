import os
import numpy as np
from skimage.io import imread
import tqdm
from concurrent.futures import ThreadPoolExecutor

# ABSOLUTE PATH OF THIS FILE
dir_path = os.path.dirname(os.path.realpath(__file__))


class DatasetManager:
    """
    The DatasetManager class has the purpose of handling and abstracting for faster use the operations that will
    have to be done on the dataset (shuffle, read, handling paths and classes, etc.).

    EXAMPLE USAGE

    dm = DatasetManager(rotational=True) # creates the object and sets it as rotational. It means that it will behave
                                         # as a circular iterator
    dm.shuffle_train()
    x, y = dm.get_batch_train_multithreaded(size=1000, workers=10)

    # x, y are respectively an array of 1000 images and an array of 1000 one-hot-encoded classes (29x1 array with a
    # single 1)


    """

    # this dictionary maps every label's name to an array index. Data are of type <x, y>, where x is the image and y
    # is a 29-elements array containing all zeros but a single 1 in the corresponding index
    labels_to_index_dict = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9, "K": 10,
                            "L": 11, "M": 12, "N": 13, "O": 14, "P": 15, "Q": 16, "R": 17, "S": 18, "T": 19, "U": 20,
                            "V": 21, "W": 22, "X": 23, "Y": 24, "Z": 25, "del": 26, "nothing": 27, "space": 28}

    # this dictionary exploits the reverse task wrt the previous dictionary. Will be useful to handle the conversion
    # of the models' numeric output to the corresponding letter in an efficient fast way.
    index_to_labels_dict = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J", 10: "K",
                            11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T", 20: "U",
                            21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z", 26: "del", 27: "nothing", 28: "space"}

    # number of classes
    n_classes = 29

    def __init__(self, path_train=None, path_test=None, rotational=False, verbose=True):
        """
        Preprocesses the images' paths and prepares them to be read in batches.
        :param path_train: path to the folder containing the training samples. If none, the path
                           ./../../resources/asl_alphabet_train will be used.
        :param path_test: path to the folder containing the test samples. If none, the path
                           ./../../resources/asl_alphabet_test will be used.
        :param rotational: if true, the get_batch_train method will always supply an output. When all the samples have
                           been used at least once, it restarts from the beginning. If false, if all the samples have
                           been used, an exception is thrown. Setting this to True is ideal when training a model.
        :param verbose:    if True, this constructor will print some informative text about what it is doing.
        """
        if path_train is None:
            self.trainpath = os.path.join(dir_path, "..", "..", "resources", "asl_alphabet_train")
        else:
            self.trainpath = path_train
        if path_test is None:
            self.testpath = os.path.join(dir_path, "..", "..", "resources", "asl_alphabet_test")
        else:
            self.testpath = path_test

        self.rotational = rotational
        self.index = 0
        if verbose:
            print("PREPARING TRAINING SAMPLES' PATHS")
        self.train_x, self.train_y = DatasetManager.__read_and_format_paths_train(self.trainpath, verbose)
        self.n_train_samples = len(self.train_x)

        if verbose:
            print("PREPARING TEST SAMPLES' PATHS")
        self.test_x, self.test_y = DatasetManager.__read_and_format_paths_test(self.testpath, verbose)
        self.n_test_samples = len(self.test_x)

    @classmethod
    def __read_and_format_paths_train(cls, path, verbose=False):
        """ Private static method that helps the constructor formatting the paths and classes' arrays."""
        x = []
        y = []
        dirs = os.listdir(path)
        for direct in dirs if not verbose else tqdm.tqdm(dirs):
            index = cls.labels_to_index_dict[direct]
            dir_abs_path = os.path.join(path, direct)
            ims = os.listdir(dir_abs_path)
            for im in ims:
                final_path = os.path.join(dir_abs_path, im)
                x.append(final_path)
                y_sample = np.zeros([cls.n_classes], dtype=np.uint8)
                y_sample[index] = 1
                y.append(y_sample)
        return np.array(x), np.array(y)

    @classmethod
    def __read_and_format_paths_test(cls, path, verbose=False):
        """ Private static method that helps the constructor formatting the paths and classes' arrays."""
        x = []
        y = []
        images = os.listdir(path)
        for im in images if not verbose else tqdm.tqdm(images):
            final_path = os.path.join(path, im)
            x.append(final_path)
            y_sample = np.zeros([cls.n_classes], dtype=np.uint8)
            y_sample[cls.labels_to_index_dict[im.split("_")[0]]] = 1
            y.append(y_sample)
        return np.array(x), np.array(y)

    def get_batch_train(self, size):
        """Single thread training batch reader.
        Works with both rotational and non-rotational DatasetManagers.
        :param size: the size of the batch to be read
        """
        if self.index == self.n_train_samples:
            raise EndOfDatasetException(EndOfDatasetException.MSG)
        x = []
        y = []
        for _ in range(size):
            x.append(imread(self.train_x[self.index]))
            y.append(self.train_y[self.index])
            self.index += 1
            if self.index == self.n_train_samples:
                if self.rotational:
                    self.index = 0
                else:
                    break
        return np.array(x), np.array(y)

    def get_batch_train_multithreaded(self, size, workers):
        """Single thread training batch reader.
        Works only with rotational DatasetManagers.
        :param size: the size of the batch to be read
        :param workers: the number of threads that will be used to split the  I/O workload
        :return a tuple (x, y) containing the requested samples and the corresponding classes.
                x will be an array of shape (size, 200, 200, 3) while y will be an array of shape (size, 29)
        """
        if not self.rotational:
            raise NotRotationalException(NotRotationalException.MSG)
        x = None
        y = None
        size_per_worker = []
        indices = []
        app = size
        spr = int(np.ceil(size/workers))
        while app > 0:
            if app > spr:
                size_per_worker.append(spr)
                app -= spr
            else:
                size_per_worker.append(app)
                app = 0
        for i in range(workers):
            f = self.index
            t = (self.index + size_per_worker[i]) % self.n_train_samples
            self.index = t
            indices.append([f, t])

        executor = ThreadPoolExecutor(max_workers=workers)
        futures = [executor.submit(self.__read_from_to, indices[i][0], indices[i][1], True)
                   for i in range(workers)]
        results = [future.result() for future in futures]
        for i in range(workers):
            if x is None:
                x = results[i][0]
                y = results[i][1]
            else:
                x = np.concatenate((x, results[i][0]))
                y = np.concatenate((y, results[i][1]))
        return np.array(x), np.array(y)

    def __read_from_to(self, f, t, train):
        """Provate method that supports the get_batch_train_multithreaded method."""
        x = []
        y = []
        i = f
        if train:
            while i != t:
                x.append(imread(self.train_x[i]))
                y.append(self.train_y[i])
                i += 1
                if i == self.n_train_samples:
                    i = 0
        else:
            while i != t:
                x.append(imread(self.test_x[i]))
                y.append(self.test_y[i])
                i += 1
                if i == self.n_test_samples:
                    i = 0
        return np.array(x), np.array(y)

    def get_test(self):
        x = []
        y = []
        for i in range(self.n_test_samples):
            x.append(imread(self.test_x[i]))
            y.append(self.test_y[i])
        return np.array(x), np.array(y)

    def shuffle_train(self):
        """
        Shuffles the elements in the train set and the corresponding classes by keeping the relative order of the pairs.
        :return: nothing
        """
        x = self.train_x
        y = self.train_y
        rx = []
        ry = []
        ids = list(range(len(x)))
        for i in range(len(x)):
            rnd = np.random.randint(0, len(ids))
            rx.append(x[ids[rnd]])
            ry.append(y[ids[rnd]])
            ids.pop(rnd)
        self.train_x = rx
        self.train_y = np.array(ry)


class EndOfDatasetException(Exception):
    MSG = "All the data were already read and this object doesn't have the 'Rotational' attribute set. Try \n" \
          "setting the attribute to true when creating the DatasetManager object. This will make the iterator \n" \
          "restart when all the images were read."


class NotRotationalException(Exception):
    MSG = "This method requires the dataset to be rotational."


if __name__ == "__main__":
    import src.data.utilities as u
    dm = DatasetManager(rotational=True)
    dm.shuffle_train()
    x, y = dm.get_batch_train_multithreaded(size=1000, workers=10)
    u.showimage(x[0])
    print(y[0])
    print(DatasetManager.index_to_labels_dict[np.argmax(y)])
    print(x.shape, y.shape)

