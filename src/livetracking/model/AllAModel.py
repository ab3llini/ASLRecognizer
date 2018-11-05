import src.livetracking.model.AbstractModel as AM
import numpy as np


class AllAModel(AM.AbstractModel):

    def predict(self, x):
        pred = np.zeros(29)
        pred[0] = 1
        return pred
