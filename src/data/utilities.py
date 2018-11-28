import matplotlib.pyplot as mplt
import numpy as np
import src.data.dataset_manager as dm

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


def create_matr_hands(n, m):
    matr = None
    data = dm.DatasetManager()
    data.shuffle_train()
    x, _ = data.get_batch_train_multithreaded(500, 10)
    index = 0
    for i in range(n):
        row = None
        for j in range(m):
            if row is None:
                row = x[index]
            else:
                row = np.hstack((row, x[index]))
            index += 1
        if matr is None:
            matr = row
        else:
            matr = np.vstack((matr, row))
    showimage(matr)
