#!/usr/bin/python3
import numpy as np

#from keras import backend as K
#K.tensorflow_backend._get_available_gpus()

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

from keras import layers
from keras import optimizers
from keras.models import Sequential
from keras.layers import recurrent
from keras.models import Model
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical

EPOCS = 10
HIDDEN_SIZE = 128
BATCH_SIZE = 32
LAYERS = 5

def addChannelShape(x):
    #return np.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
    return np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 1))

def trainModel(X_train, y_train, X_test, y_test):
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    numSamples = X_train.shape[0] #len(X_train)
    numWindows = X_train.shape[1]
    numFreqs = X_train.shape[2]
    print(np.unique(y_train))
    numClasses = np.unique(y_train).shape[0]
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print(y_test)
    print("Samples: " + str(numSamples) + "\nWindows: " + str(numWindows) + "\nFrequency windows: " + str(numFreqs))
    X_train = addChannelShape(X_train)
    X_test = addChannelShape(X_test)
    model = Sequential()
    HIDDEN_SIZE = numFreqs
    model.add(layers.GaussianDropout(0.20))
    #model.add(layers.GaussianNoise(0.50))
    #model.add(layers.Conv2D(32, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape=(X_train.shape[1],X_train.shape[2],1)))#, data_format="channels_last"))
    model.add(layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), activation='relu', input_shape=(X_train.shape[1],X_train.shape[2],1), data_format="channels_last"))
    model.add(layers.Conv2D(64, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(64, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(layers.Conv2D(64, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dense(numClasses, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.01), metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Training")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCS, batch_size=64)
    print(model.summary())
    print('Evaluation')
    loss, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print("Loss: " + str(loss))
    print("Accuracy: " + str(acc))
    y_pred = model.predict(X_test)
    matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(matrix)
