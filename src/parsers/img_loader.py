from keras.preprocessing import image
import sklearn.utils as skutils
from enum import Enum
import os
import numpy as np
import psutil
from threading import Thread
import math
from matplotlib import pyplot as plt


#################################################################################
#                                 README                                        #
#################################################################################
# Please make sure to have the two folders asl_alphabet_test & asl_alphabet_train
# under a parent directory named dataset in the root of the project
# It will not be synchronized to github due to its size (approx 1GB)

# Default parameters. Do not edit unless sure of what you are doing.
bpath_def = '../../dataset/'
trpath_def = 'asl_alphabet_train/'
tspath_def = 'asl_alphabet_test/'
classes_def = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']


class DatasetParserMode(Enum):
    """
    This enumeration is used to provide elegant logic in the parser and iterator implementations
    You should not make direct use of it until you are absolutely sure of what you are doing
    """
    PATH_BASED = 0
    CLASS_BASED = 1


class DatasetParser:
    """
    This class provides parsing functionality for the dataset
    It is capable of splitting into training and testing set the available data in a very fast and efficient way
    It has two modes of operation: single thread and multi thread
    We strongly advise you to run it in default mode (multi thread) due to performance reasons
    In single core it will take a long time since JPEG -> RGB takes time and creates a lot of overhead (approx 10x)
    1GB of data will be stored in more than 10GB of RAM & SWAP File if multi threading is left off.

    Note 1: Multi threading will provide a HUGE boost, use it.
    Note 2: The parser will spawn a number of threads equal to the number of logical CPUs available on your machine

    :param multithread: Defines how the parser operates, we advise to leave this parameter to on
    :param maxthreads: Specify the max number of threads that will be created
    :param basepath: Leave default value unless you want to feed a different dataset base URL
    :param trainingpath: Training set URL relative to basepath.
    Leave default value unless you want to feed a different training dataset URL
    :param testingpath: Testing set URL relative to basepath.
    Leave default value unless you want to feed a different testing base URL
    :param verbose: I hope you understood what this is here for :)

    """
    def __init__(self,
                 multithread=True,
                 maxthreads=psutil.cpu_count(),
                 basepath=bpath_def,
                 trainingpath=trpath_def,
                 testingpath=tspath_def,
                 verbose=True,
                 ):
        self.tr_path = basepath + trainingpath
        self.ts_path = basepath + testingpath
        self.multithread = multithread
        self.maxthreads = maxthreads
        self.verbose = verbose

    # This method creates objects to handle and operate the multi threaded parsing infrastructure.
    # It should be called just by the class itself
    def _multithread_splitter(self, n):
        nthreads = min(n, self.maxthreads)
        work_size = math.floor(n / nthreads)
        threads = [None] * nthreads
        results = [[[], []] for _ in range(nthreads)]
        if self.verbose:
            print('Multithread parsing is on.')
            print('Available number of logical CPUs = %s, maximum number of threads set to = %s'
                  % (psutil.cpu_count(), self.maxthreads))

        return nthreads, work_size, threads, results

    # This method fetches a set of images from a set of class labels.
    # This method is not used by the iterator since the fetch is not based on URLs but on class names.
    def _subroutine_fetch_tr(self, selected, results=None):

        x = results[0] if self.multithread else []
        y = results[1] if self.multithread else []

        for c in selected:
            for sample in os.listdir(self.tr_path + c):
                path = self.tr_path + c + '/' + sample
                img_vect = self.img2array(path)
                x.append(img_vect)
                y.append([1 if c_ == c else 0 for c_ in classes_def])

            if self.verbose:
                print('Update: %s parsed' % c)

        return x, y

    # This method fetches a set of images from a set of URLs, usually provided by the iterator class.
    def _subroutine_fetch_paths(self, paths, classes, results=None):

        x = results[0]
        y = results[1]

        for idx, path in enumerate(paths):
            img_vect = self.img2array(path)
            x.append(img_vect)
            y.append([1 if c_ == classes[idx] else 0 for c_ in classes_def])

        return x, y if self.multithread else None

    def fetch_tr(self, mode=DatasetParserMode.CLASS_BASED, *args):
        """
        This method creates the vector representation of the dataset.
        It is modular in that it can be called directly by the user or by the iterator
        Without an iterator class, this method will merely return all the parsed images from the specified classes
        No shuffling will be performed when called directly

        Direct call by the user
        :param mode: should always be set to DatasetParserMode.CLASS_BASED.
        :param args: A sequence of classes that will make up the final result vector,
        if set to, say, "A", "B", "C", the method will return images and labels for those classes only. (No shuffle)

        Call by the iterator
        Do not try to emulate the iterator with a direct call unless sure of what you are doing
        :param mode: should always be set to DatasetParserMode.PATH_BASED.
        :param args: A pair of paths/classes that will be parsed and returned in the final result

        :return x: The vector representation of all the images, each has shape 200x200x3. x shape varies
        :return y: The vector representation of all the classes corresponding to the images. y shape varies

        """
        selected = []

        if mode == DatasetParserMode.PATH_BASED:

            nthreads, work_size, threads, results = self._multithread_splitter(len(args[0][1]))
            if self.verbose:
                print('Splitting load across %s threads..' % nthreads)
            for idx in range(nthreads):

                if idx + 1 != nthreads:
                    paths_arg = args[0][0][idx * work_size: idx * work_size + work_size]
                    classes_arg = args[0][1][idx * work_size: idx * work_size + work_size]
                else:
                    paths_arg = args[0][0][idx * work_size:]
                    classes_arg = args[0][1][idx * work_size:]

                if self.verbose:
                    print('Thread %s -> %s paths' % (idx, len(paths_arg)))

                threads[idx] = Thread(target=self._subroutine_fetch_paths, args=(paths_arg, classes_arg, results[idx]))
                threads[idx].start()

            if self.verbose:
                print('Suspending main thread until all other threads have finished.')
            for idx in range(len(threads)):
                threads[idx].join()
                if self.verbose:
                    print('Thread %s has done parsing..' % idx)

            if self.verbose:
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

                if self.verbose:
                    print('Splitting load across %s threads..' % nthreads)

                for idx in range(nthreads):

                    if idx + 1 != nthreads:
                        arg = selected[idx * work_size: idx * work_size + work_size]
                    else:
                        arg = selected[idx * work_size:]

                    if self.verbose:
                        print('Thread %s -> %s' % (idx, arg))

                    threads[idx] = Thread(target=self._subroutine_fetch_tr, args=(arg, results[idx]))
                    threads[idx].start()

                if self.verbose:
                    print('Suspending main thread until all other threads have finished.')
                for idx in range(len(threads)):
                    threads[idx].join()
                    if self.verbose:
                        print('Thread %s has finished.' % idx)

                if self.verbose:
                    print('All threads have finished parsing.. Combining results..')

                x, y = [], []

                for result in results:
                    for i in range(len(result[0])):
                        x.append(result[0][i])
                        y.append(result[1][i])

                return np.array(x), np.array(y)

            else:
                if self.verbose:
                    print('Multithread parsing is off')
                x, y = self._subroutine_fetch_tr(selected=selected)
                return np.array(x), np.array(y)

    # Converts an image into a numpy RBG array 200x200x3
    @staticmethod
    def img2array(path):
        img = image.load_img(path)
        return image.img_to_array(img)


class TrainingSetIterator:

    """
    This class provides a simple way to load the dataset incrementally without
    filling up your ram/swap file with 10GB of data
    This happens because the images originally are in JPEG but after decompression in RGB the increase their size.
    HOW TO USE:
    1) Instantiate specifying the used parser and the batchsize
    2) Simply use it like: for x,y in instance: operate on the current batch
    :param parser: Valid instance of a DatasetParser with multithread = true
    :param batchsize: The size of each batch. The total training set should be a multiple of this number since each batch
    should have the same size.
    :param shuffle: Set it to true to shuffle the images returned in each batch
    :param seed: If you want always the same batches, provide a seed for the initialization of the random algorithm
    :param classes: leave empty to create batches from all the training classes, or specify your own by passing a list
    """

    def __init__(self, parser, batchsize=1000, shuffle=False, seed=None, classes=None):
        self.parser = parser
        self.processed = 0
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.seed = seed
        self.classes = classes if classes is not None and len(classes) > 0 else classes_def
        self.x, self.y = self._fetch_image_data()

    # Returns the iterator for this class. You shouldn't require to call this
    def get_iterator(self):
        return iter(self)

    # Fetches all the paths for each training images present in the dataset
    # These paths will later be passed to the parser for the creation of the batch
    # This method will take care of shuffling the image path and, in turn, the final batch
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

        if self.shuffle:
            x, y = skutils.shuffle(x, y, random_state=self.seed)

        if self.nimages % self.batchsize != 0:
            raise Exception('The training set size is not a multiple of the'
                            ' batch size! Batch size can only be one of these values:\n%s'
                            % self.available_batchsizes(self.nimages)
                            )

        return x, y

    # This method should not be called directly but by the class itself only.
    # It creates the next batch of path/class pairs that will be processed by the parser
    def _get_next_batch(self):
        ret_x, ret_y = [], []
        for _ in range(self.batchsize):
            ret_x.append(self.x.pop())
            ret_y.append(self.y.pop())

        return ret_x, ret_y

    # This method is required to implement the iterator interface
    def __iter__(self):
        return self

    # This method is required to implement the iterator interface
    # It will provide the next batch as a x/y pair of numpy arrays
    # It has been thought to be used withing for loops as specified in the class definition
    # This method will pass the current batch of url/class pairs to the parser and will
    #  return the image/class vector pairs
    def __next__(self):
        if self.processed < self.nimages:

            self.currentbatch = self._get_next_batch()
            self.currentbatch = self.parser.fetch_tr(DatasetParserMode.PATH_BASED,
                                                     [self.currentbatch[0], self.currentbatch[1]])
            self.processed += self.batchsize
            return self.currentbatch
        else:
            raise StopIteration

    @staticmethod
    def available_batchsizes(x):
        # This function takes a number and prints the factors
        v = ''
        for i in range(1, x + 1):
            if x % i == 0:
                v += str(i) + '\n'
        return v


"""
After various test, batch sizes around 4350 should be fine, 
otherwise keep it less than of equal to
"""
ds_parser = DatasetParser(verbose=True)
training_set = TrainingSetIterator(parser=ds_parser, shuffle=True, batchsize=3625, seed=None)

for x, y in training_set:
    print('Printing the first image of this new batch..')
    plt.imshow(x[0]/255.)
    print('The class vector corresponding to the first image is..')
    print(y[0])
    plt.show()
