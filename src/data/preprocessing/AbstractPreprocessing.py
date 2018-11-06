import abc
import numpy as np


class AbstractPreprocessing:

    def preprocess_array(self, x):
        result = []
        for samp in x:
            result.append(self.preprocess(samp))
        return np.array(result)

    @abc.abstractmethod
    def preprocess(self, x):
        pass
