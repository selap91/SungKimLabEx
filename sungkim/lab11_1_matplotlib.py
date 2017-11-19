import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

sess = tf.InteractiveSession()
image = np.array([[[[1], [2], [3]], [[4],[5],[6]], [[7],[8],[9]]]], dtype=np.float32)

print(image.shape)
plt.imshow(image.reshape(3,3), cmap="Greys")
plt.show()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

img = mnist.train.images[100].reshape(28,28)
plt.imshow(img, cmap="gray")  # Greys : 흰바탕 검은글씨 , gray : 검은바탕 흰글씨
plt.show()