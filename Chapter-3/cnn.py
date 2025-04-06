# Imports
import tensorflow as tf
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard


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
training_images=  training_images.reshape(60000,28,28,1) # reshaping to match the input layer of the Convolution network input layer.
training_images = training_images / 255.0

test_images=  test_images.reshape(10000,28,28,1)
test_images = test_images / 255.0

#Define how your model should be.
model = Sequential([
                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                    tf.keras.layers.MaxPooling2D((2, 2)),  # Fixed typo
                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D(2,2),
                    tf.keras.layers.Flatten(),  # Removed input_shape
                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

#compile
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#fit
model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])

#evaluate
model.evaluate(test_images, test_labels)
model.summary()         
