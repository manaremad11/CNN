import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses

# load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# data parameters
num_classes = 10
input_shape = (28, 28, 1)
# scale images to [0-1]
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# make images in dimension (28, 28, 1) (width, height, depth)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
# convert labels to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential([
      keras.Input(shape=input_shape),
      layers.Conv2D(32, kernel_size=(5, 5), activation="relu"),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Conv2D(64, kernel_size=(5, 5), activation="relu"),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Flatten(),
      layers.Dense(num_classes, activation="softmax"),
])

model.compile(loss=losses.CategoricalCrossentropy(),
              optimizer=optimizers.SGD(learning_rate=0.001, momentum=0.9),
              metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=128, epochs=1, validation_split=0.1,
          shuffle=True,)
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])