import matplotlib.pyplot as mplt
import numpy as np


def showimage(image):
    """displays a single image in SciView"""
    mplt.figure()
    mplt.imshow(image)
    mplt.show()


def shuffle2(x, y):
    rx = []
    ry = []
    ids = list(range(len(x)))
    for i in range(len(x)):
        rnd = np.random.randint(0, len(ids))
        rx.append(x[ids[rnd]])
        ry.append(y[ids[rnd]])
        ids.pop(rnd)
    return np.array(rx), np.array(ry)