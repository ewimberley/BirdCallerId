#!/usr/bin/python3
import numpy as np

from keras import layers
from keras.models import Sequential
from keras.layers import recurrent
from keras.models import Model
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical

# Try replacing GRU, or SimpleRNN.
RNN = layers.LSTM
EPOCS = 5
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 2

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
    model = Sequential()
    model.add(RNN(numFreqs, return_sequences=True))
    for _ in range(LAYERS-1):
        model.add(RNN(numFreqs))
    model.add(layers.Dense(numClasses, activation='sigmoid'))
    #model.add(layers.TimeDistributed(layers.Dense(len(X_train), activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
