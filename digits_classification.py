import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# scaling the data around 0 and 1 to increase accuracy
x_train = x_train / 255
x_test = x_test / 255

# turning data from 2d array to 1d array
x_train_flat = x_train.reshape(len(x_train), 28*28)
x_test_flat = x_test.reshape(len(x_test), 28*28)

# sequential means having a stack of neural network
# since it is a stack it will take every layer as in input
# it has two layers one is input other is output
# output layer with 10 features
# input layer with 784 features
# keras has api layers.Dense
# Dense ensures that every neuron in one layer is
# connected with the neuron in the second layer
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

# compiling neural network
# passing bunch of parameters
# optimizer --> help you train efficiently
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train_flat, y_train, epochs=5)

# before uploading a model to a production
# we need to do testing
# model.evaluate(x_test_flat, y_test)

# predicting a number
y_predicted = model.predict(x_test_flat)
print(y_predicted)
print(np.argmax(y_predicted[1]))

# confusion matrix
# basically it helps in identify how many times neural network
# prediction was correct

y_predicted_labels = [np.argmax(i) for i in y_predicted]
print(y_predicted_labels[:10])
print(y_predicted[:10])

cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
print(cm)

# printing confusion matrix in a visually appealing way

import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('predicted')
plt.ylabel('truth ')

plt.show()
