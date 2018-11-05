import src.livetracking.model.AbstractModel as AM
import numpy as np


class RandomModel(AM.AbstractModel):

    def predict(self, x):
        pred = np.zeros(29)
        pred[np.random.randint(0, 29)] = 1
        return pred
