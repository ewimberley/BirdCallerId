#!/usr/bin/python3
import numpy as np

from keras import layers
from keras.layers import recurrent
from keras.models import Model

# Try replacing GRU, or SimpleRNN.
RNN = layers.LSTM
EPOCS = 5
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1

def trainModel(X_train, y_train, X_test, y_test):
    model = Sequential()
    for _ in range(LAYERS):
        model.add(RNN(HIDDEN_SIZE))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCS, batch_size=64)
