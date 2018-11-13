from src.gan.models import cnn
from src.data import dataset_manager, utilities
import numpy as np
import keras


dm = dataset_manager.DatasetManager()
dm.shuffle_train()
size = 1000
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

opt_discr = keras.optimizers.Adam(lr=1e-3)
loss = keras.losses.binary_crossentropy
opt_gen = keras.optimizers.Adam(lr=1e-3)

discr.compile(opt_discr, loss, metrics=['accuracy'])
print(len(discr.layers))
discr.summary()


while True:
    print("DISCRIMINANT")
    discr.fit(x=x, y=y, batch_size=50, epochs=1)
    net = cnn.merge_generator_discr(gen, discr)
    print("NETS ATTACHED")
    net.compile(opt_gen, loss, metrics=['accuracy'])
    net.summary()
    net.fit(x=input_noise, y=output_noise, epochs=1, batch_size=10)
    gen = cnn.recreate_gen(net)
    pred = np.floor(gen.predict(input_noise[0]).squeeze())
    pred = np.array(pred, np.uint8)
    utilities.showimage(pred)




