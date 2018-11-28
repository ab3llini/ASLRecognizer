import src.livetracking.model.AbstractModel as AM
import cv2
import numpy as np
import src.data.utilities as u


class HarmonicKerasModel(AM.AbstractModel):

    def __init__(self, model, memory):
        super().__init__(model)
        self.memory = memory
        self.prediction = np.zeros(29)

    def predict(self, x):
        res = x[140:340, 220:420, :]
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        pred = self.model.predict(np.expand_dims(res, axis=0))
        self.prediction = (1/self.memory)*pred + self.prediction*(1 - 1/self.memory)
        return self.prediction
