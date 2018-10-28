from keras.preprocessing import image
from sklearn.utils import shuffle
from enum import Enum
import os
import numpy as np
import psutil
from threading import Thread
import math
from matplotlib import pyplot as plt

# Please make sure to have the two folders asl_alphabet_test & asl_alphabet_train
# under a parent directory named dataset in the root of the project
# It will not be synchronized to github due to its size (approx 1GB)


bpath_def = '../../dataset/'
trpath_def = 'asl_alphabet_train/'
tspath_def = 'asl_alphabet_test/'
classes_def = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']


class DatasetParserMode(Enum):
    PATH_BASED = 0
    CLASS_BASED = 1


# Parses the dataset into training and testing set and performs various operations on the data
class DatasetParser:

    # Initialize the parser.
    # You can pass different paths if you plan to use a different/modified dataset in another location
    # Note that reading the whole dataset using just one core is extremely slow.
    # This parser is multithread-ready
    def __init__(self, multithread=True, basepath=bpath_def, trainingpath=trpath_def, testingpath=tspath_def):
        self.tr_path = basepath + trainingpath
        self.ts_path = basepath + testingpath
        self.multithread = multithread

    # Init multithread parsing infrastructure
    @staticmethod
    def _multithread_splitter(n):
        nthreads = min(psutil.cpu_count(), n)
        work_size = math.floor(n / nthreads)
        threads = [None] * nthreads
        results = [[[], []] for _ in range(nthreads)]

        print('Multithread parsing is on.')
        print('Available number of logical CPUs = %s' % psutil.cpu_count())

        return nthreads, work_size, threads, results

    # This is a subroutine and should only be called by the parser itself.
    # Do not make edits to it
    def _subroutine_fetch_tr(self, selected, results=None):

        x = results[0] if self.multithread else []
        y = results[1] if self.multithread else []

        for c in selected:
            for sample in os.listdir(self.tr_path + c):
                path = self.tr_path + c + '/' + sample
                img_vect = self.img2array(path)
                x.append(img_vect)
                y.append([1 if c_ == c else 0 for c_ in classes_def])

            print('Update: %s parsed' % c)

        return x, y

    # This is a subroutine and should only be called by the parser itself.
    # Do not make edits to it
    def _subroutine_fetch_paths(self, paths, classes, results=None):

        x = results[0]
        y = results[1]

        for idx, path in enumerate(paths):
            img_vect = self.img2array(path)
            x.append(img_vect)
            y.append([1 if c_ == classes[idx] else 0 for c_ in classes_def])

        return x, y if self.multithread else None

    # Returns two vectors x and y
    # x contains the image vector representations as a 200x200x3 matrix
    # y contains a
    def fetch_tr(self, mode=DatasetParserMode.CLASS_BASED, *args):

        selected = []

        if mode == DatasetParserMode.PATH_BASED:

            nthreads, work_size, threads, results = self._multithread_splitter(len(args[0][1]))
            print('Splitting load across %s threads..' % nthreads)
            for idx in range(nthreads):

                if idx + 1 != nthreads:
                    paths_arg = args[0][0][idx * work_size: idx * work_size + work_size]
                    classes_arg = args[0][1][idx * work_size: idx * work_size + work_size]
                else:
                    paths_arg = args[0][0][idx * work_size:]
                    classes_arg = args[0][1][idx * work_size:]

                print('Thread %s -> %s paths' % (idx, len(paths_arg)))

                threads[idx] = Thread(target=self._subroutine_fetch_paths, args=(paths_arg, classes_arg, results[idx]))
                threads[idx].start()

            print('Suspending main thread until all other threads have finished.')
            for idx in range(len(threads)):
                threads[idx].join()
                print('Thread %s has done parsing..' % idx)

            print('All threads have finished parsing.. Combining results..')

            x, y = [], []

            for result in results:
                for i in range(len(result[0])):
                    x.append(result[0][i])
                    y.append(result[1][i])

            return np.array(x), np.array(y)

        else:
            # Check user input and update selection
            if len(args) > 0:
                for c in args:
                    selected.append(c.upper()) if len(c) == 1 else selected.append(c)
            else:
                selected = classes_def

            if self.multithread:

                nthreads, work_size, threads, results = self._multithread_splitter(len(selected))

                print('Splitting load across %s threads..' % nthreads)

                for idx in range(nthreads):

                    if idx + 1 != nthreads:
                        arg = selected[idx * work_size: idx * work_size + work_size]
                    else:
                        arg = selected[idx * work_size:]

                    print('Thread %s -> %s' % (idx, arg))

                    threads[idx] = Thread(target=self._subroutine_fetch_tr, args=(arg, results[idx]))
                    threads[idx].start()

                print('Suspending main thread until all other threads have finished.')
                for idx in range(len(threads)):
                    threads[idx].join()
                    print('Thread %s has finished.' % idx)

                print('All threads have finished parsing.. Combining results..')

                x, y = [], []

                for result in results:
                    for i in range(len(result[0])):
                        x.append(result[0][i])
                        y.append(result[1][i])

                return np.array(x), np.array(y)

            else:
                print('Multithread parsing is off')
                x, y = self._subroutine_fetch_tr(selected=selected)
                return np.array(x), np.array(y)

    @staticmethod
    def img2array(path):
        img = image.load_img(path)
        return image.img_to_array(img)


class TrainingSetIterator:

    def __init__(self, parser, batchsize=1000, seed=None, classes=None):
        self.parser = parser
        self.processed = 0
        self.batchsize = batchsize
        self.seed = seed
        self.classes = classes if classes is not None and len(classes) > 0 else classes_def
        self.x, self.y = self._fetch_image_data()

    def get_iterator(self):
        return iter(self)

    def _fetch_image_data(self):

        x = []
        y = []

        print('Collecting image paths and classes..')
        for c in self.classes:

            elements = os.listdir(bpath_def + trpath_def + c)
            l = len(elements)
            x += [bpath_def + trpath_def + c + '/' + e for e in elements]
            y += [c for _ in range(l)]

        self.nimages = len(x)

        print('Found %s images' % self.nimages)
        print('Shuffling images..')

        x, y = shuffle(x, y, random_state=self.seed)

        if self.nimages % self.batchsize != 0:
            raise Exception('The training set size is not a multiple of the batch size!')

        return x, y

    def _get_next_batch(self):
        ret_x, ret_y = [], []
        for _ in range(self.batchsize):
            ret_x.append(self.x.pop())
            ret_y.append(self.y.pop())

        return ret_x, ret_y

    def __iter__(self):
        return self

    def __next__(self):
        if self.processed < self.nimages:

            self.currentbatch = self._get_next_batch()
            self.currentbatch = self.parser.fetch_tr(DatasetParserMode.PATH_BASED,
                                                     [self.currentbatch[0], self.currentbatch[1]])
            self.processed += self.batchsize
            return self.currentbatch
        else:
            raise StopIteration


ds_parser = DatasetParser()
training_set = TrainingSetIterator(parser=ds_parser, batchsize=1000, seed=None)

for x, y in training_set:
    print('New batch..')
    plt.imshow(x[0]/255.)
    print(y[0])
    plt.show()