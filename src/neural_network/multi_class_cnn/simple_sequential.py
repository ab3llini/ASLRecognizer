from src.data.dataset_manager import DatasetManager
from src.neural_network.trainer.KerasModel import Kerasmodel
from src.models import cnn
import keras


if __name__ == '__main__':
    # load
    load = True
    # learning rate
    lr = 0.5e-2
    # l2 regularization coefficient
    wd = 1e-9
    # dataset creation
    data = DatasetManager()
    # shuffle train set to give similar variance to each batch
    data.shuffle_train()
    # get tiny test data
    test_x, test_y = data.get_test()

    # create keras model
    if load:
        model = keras.models.load_model("./../../../resources/keras_saves/matteo_simple_thin.h5")
    else:
        model = cnn.thin_sequential(wd_rate=wd)
    trainer = Kerasmodel(train_set_provider=data,
                         model=model,
                         opt=keras.optimizers.SGD(lr=lr))
    trainer.train(iterations=5,
                  epochs_per_iteration=9,
                  validation=(test_x, test_y),
                  sleep_time=3*60,
                  rest_every=6,
                  train_batch_size=50,
                  train_set_chunk_size=2000)
    trainer.save("./../../../resources/keras_saves/matteo_simple_thin.h5")

