from src.gan.models import cnn
from src.data import dataset_manager, utilities
import numpy as np
import keras


dm = dataset_manager.DatasetManager()
dm.shuffle_train()
size = 100
x, _ = dm.get_batch_train_multithreaded(size=size, workers=8)
y = np.ones(size)

input_noise = np.random.normal(size=[size, 200, 200, 1])
output_noise = np.ones(size)

x_noise = np.random.normal(size=[size, 200, 200, 3])
y_noise = np.zeros(size)

x = np.concatenate((x, x_noise))
y = np.concatenate((y, y_noise))

x, y = utilities.shuffle2(x, y)

discr = cnn.discriminator()
gen = cnn.generator()

opt_discr = keras.optimizers.Adam(lr=1e-5)
loss = keras.losses.binary_crossentropy
opt_gen = keras.optimizers.Adam(lr=1e-6)

discr.compile(opt_discr, loss, metrics=['accuracy'])
print(len(discr.layers))
discr.summary()


while True:
    print("TRAINING DISCRIMINANT")
    discr.fit(x=x, y=y, batch_size=50, epochs=3)
    net = cnn.merge_generator_discr(gen, discr)
    print("NETS ATTACHED")
    net.compile(opt_gen, loss, metrics=['accuracy'])
    net.fit(x=input_noise, y=output_noise, epochs=10, batch_size=10)
    gen = cnn.recreate_gen(net)
    print("GENERATOR DETACHED")
    some_predictions = gen.predict(input_noise[:100])
    some_predictions[some_predictions > 255] = 255
    some_predictions[some_predictions < 0] = 0
    some_predictions = np.floor(some_predictions).squeeze()
    pred = np.array(some_predictions, np.uint8)
    y_new = np.zeros(100)
    utilities.showimage(pred[0])
    x = np.concatenate((x, pred))
    y = np.concatenate((y, y_new))




