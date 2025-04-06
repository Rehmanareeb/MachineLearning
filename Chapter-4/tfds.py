import tensorflow as tf 
import tensorflow_datasets as tfds 

msint_data = tfds.load('fashion_mnist',split='train')

assert isinstance(msint_data,tf.data.Dataset)
for item in msint_data.take(1):
    print(item.keys())
    print(item['image'])