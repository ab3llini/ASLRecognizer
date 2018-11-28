from src.data.preprocessing.AbstractPreprocessing import AbstractPreprocessing
from skimage.transform import resize


class Resize5050(AbstractPreprocessing):

    def preprocess(self, x):
        return resize(x, [50, 50])
