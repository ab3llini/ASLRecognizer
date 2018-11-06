from src.data.preprocessing.AbstractPreprocessing import AbstractPreprocessing


class NoPreprocessing(AbstractPreprocessing):

    def preprocess(self, x):
        return x