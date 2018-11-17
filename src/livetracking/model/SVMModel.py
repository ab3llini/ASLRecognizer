import src.livetracking.model.AbstractModel as AM
import numpy as np
import cv2


class SVMModel(AM.AbstractModel):

    def predict(self, x):
        res = x[140:340, 220:420, :]
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        return self.model.predict(np.expand_dims(res, axis=0))
