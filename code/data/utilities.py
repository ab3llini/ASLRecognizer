import matplotlib.pyplot as mplt


def showimage(image):
    """displays a single image in SciView"""
    mplt.figure()
    mplt.imshow(image)
    mplt.show()