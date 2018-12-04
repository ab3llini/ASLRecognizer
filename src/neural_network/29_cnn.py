import numpy as np
import keras
from src.data.dataset_manager import DatasetManager
from src.models import cnns_29model
from src.neural_network.trainer.KerasModel1 import Kerasmodel1

if __name__ == '__main__':

    load = False
    dm  = DatasetManager()
    dm.shuffle_train()
    test_x, test_y = dm.get_test()
    print(np.shape(test_y[:,0]))
    # create keras model
    if load:
        model = keras.models.load_model("./../../resources/keras_saves/yats_test.h5")
    else:
        model = cnns_29model.cnn2(penalty=1e-8)
    trainer = Kerasmodel1(train_set_provider=dm,
                         model=model,
                         opt=keras.optimizers.adam(lr=1e-3))
    for i in range(29):
        trainer.train(iterations=6,
                  epochs_per_iteration=2,
                  validation=(test_x/255, test_y[:,i]),
                  sleep_time=3 * 60,
                  rest_every=6,
                  train_batch_size = 20,
                  train_set_chunk_size = 2000,
                  letter=i,
                  class_weight={1:26, 0:3})
        print(trainer.predict(test_x/255))
        trainer.save("./../../resources/keras_saves/yats_test4" + DatasetManager.index_to_labels_dict[i] + ".h5")
