from src.data.dataset_manager import DatasetManager
from src.neural_network.trainer.KerasModel import Kerasmodel
from src.models import cnn
import keras
from src.data.preprocessing.Resize5050 import Resize5050


if __name__ == '__main__':
    # load
    load = False
    # learning rate
    lr = 0.5e-6
    # l2 regularization coefficient
    wd = 1e-5
    # dataset creation
    data = DatasetManager(preprocessing=Resize5050())
    # shuffle train set to give similar variance to each batch
    data.shuffle_train()
    # get tiny test data
    test_x, test_y = data.get_test()

    # create keras model
    if load:
        model = keras.models.load_model("./../../../resources/keras_saves/matteo_simple_thin_res5050.h5")
    else:
        model = cnn.thin_sequential_rgb_5050(wd_rate=wd)
    trainer = Kerasmodel(train_set_provider=data,
                         model=model,
                         opt=keras.optimizers.SGD(lr=lr))
    trainer.train(iterations=20,
                  epochs_per_iteration=2,
                  validation=(test_x, test_y),
                  sleep_time=3*60,
                  rest_every=6,
                  train_batch_size=200,
                  train_set_chunk_size=5000)
    trainer.save("./../../../resources/keras_saves/matteo_simple_thin_res5050.h5")

