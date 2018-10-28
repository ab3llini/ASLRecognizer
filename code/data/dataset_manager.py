import os
import numpy as np
from skimage.io import imread


dir_path = os.path.dirname(os.path.realpath(__file__))


class DatasetManager:

    labels_to_index_dict = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9, "K": 10,
                            "L": 11, "M": 12, "N": 13, "O": 14, "P": 15, "Q": 16, "R": 17, "S": 18, "T": 19, "U": 20,
                            "V": 21, "W": 22, "X": 23, "Y": 24, "Z": 25, "del": 26, "nothing": 27, "space": 28}

    index_to_labels_dict = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J", 10: "K",
                            11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T", 20: "U",
                            21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z", 26: "del", 27: "nothing", 28: "space"}

    n_classes = 29

    def __init__(self, path_train=None, path_test=None, rotational=False):
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
        self.train_x, self.train_y = DatasetManager.__read_and_format_paths_train(self.trainpath)
        self.n_train_samples = len(self.train_x)

        #self.test_x, self.test_y = DatasetManager.__read_and_format(self.testpath)

    @classmethod
    def __read_and_format_paths_train(cls, path, test=False):
        x = []
        y = []
        if not test:
            dirs = os.listdir(path)
            for direct in dirs:
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

    def get_batch(self, size):
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

    def shuffle_train(self):
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


if __name__ == "__main__":
    import code.data.utilities as u
    dm = DatasetManager()
    dm.shuffle_train()
    x, y = dm.get_batch(1000)
    u.showimage(x[0])
    print(y[0])
    print(x.shape, y.shape)
