import numpy as np
import src.livetracking.model.AllAModel as AllAModel
import src.livetracking.model.RandomModel as RandomModel
import src.livetracking.model.KerasModel as KerasModel
import cv2
import keras
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from src.data.dataset_manager import DatasetManager

# to change
model = KerasModel.KerasModel(keras.models.load_model("./../../resources/keras_saves/matteo_simple_thin.h5"))
classes = DatasetManager.classes_list_single_letter
cap = cv2.VideoCapture(0)
shape = None
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        if shape is None:
            shape = np.shape(frame)
            bars_positions = np.floor(np.linspace(0, shape[1], 29)).astype(np.int32)

        cv2.rectangle(frame, (220, 140), (420, 340), (255, 255, 00), 2)
        pred = model.predict(frame).squeeze() * 10

        fig = Figure()

        fig.gca().bar(classes, pred, zorder=1, align='center')
        fig.gca().imshow(frame, zorder=0, extent=[0, 32, 0, 24])
        canvas = FigureCanvas(fig)
        canvas.draw()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(shape)
        cv2.imshow('frame', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()