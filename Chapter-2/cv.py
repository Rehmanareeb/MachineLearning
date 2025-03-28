# Imports
import tensorflow as tf
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Callbacks - stops training at a particular accuracy percentage to avoid overfitting or after a certain event.

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if (logs.get('accuracy')  > 0.95):
      print("\n Reached 95% accuracy so cancelling training")
      self.model.stop_training = True


callbacks = myCallback()

#Data
data = tf.keras.datasets.fashion_mnist

#Train and Test split
(training_images, training_labels), (test_images, test_labels) = data.load_data()

#Normalize images (0 - 255)
training_images = training_images / 255.0
test_images = test_images / 255.0

#Define how your model should be.
model = Sequential([tf.keras.layers.Flatten(input_shape = (28, 28)),
                    tf.keras.layers.Dense(128, activation = tf.nn.relu),
                    tf.keras.layers.Dense(10, activation = tf.nn.softmax)])

#compile
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#fit
model.fit(training_images, training_labels, epochs = 50, callbacks=[callbacks])

#evaluate
model.evaluate(test_images, test_labels)
