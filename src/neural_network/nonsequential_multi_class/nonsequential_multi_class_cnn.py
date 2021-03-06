from src.data.dataset_manager import DatasetManager
from src.neural_network.trainer.KerasModel import Kerasmodel
from src.models import cnn
import keras


if __name__ == '__main__':
    # load
    load = True
    # learning rate
    lr = 1e-6
    wd = 1e-3  # l2 regularization coefficient
    # dataset creation
    data = DatasetManager()
    # shuffle train set to give similar variance to each batch
    data.shuffle_train()
    data.shuffle_test()
    # get tiny test data
    test_x, test_y = data.get_test()

    # create keras model
    if load:
        model = keras.models.load_model("./../../../resources/keras_saves/nonseq3_20.h5")
    else:
        model = cnn.non_seq_new2(wd_rate=wd, num=20)
    trainer = Kerasmodel(train_set_provider=data,
                         model=model,
                         opt=keras.optimizers.Adam(lr=lr))
    trainer.train(iterations=0,
                  epochs_per_iteration=3,
                  validation=(test_x, test_y),
                  sleep_time=3*60,
                  rest_every=20,
                  train_batch_size=10,
                  train_set_chunk_size=2000,
                  augment=False)
    trainer.evaluate(test_x, test_y)
    # trainer.save("./../../../resources/keras_saves/nonseq3_20.h5")
