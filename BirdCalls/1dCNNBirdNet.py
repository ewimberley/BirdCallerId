#!/usr/bin/python3
#from birdnetRNN import *
from BirdCalls.birdnetSignalProcessing import *
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

from BirdCalls.birdnetSignalProcessing import wavFileToNpy, sampleWindows

X_train = []
y_train = []
X_validate = []
y_validate = []
X_test = []
y_test = []

def addChannelShape(x):
    return np.reshape(x, (x.shape[0], x.shape[1], 1))

def getDatasetArrays(dataset):
    xArray = X_train
    yArray = y_train
    if dataset == "validate":
        xArray = X_validate
        yArray = y_validate
    elif dataset == "test":
        xArray = X_test
        yArray = y_test
    return xArray, yArray

def computeSpeciesSamplingRatio(df, datasetName):
    speciesToTime = {}
    speciesSamplingRatio = {}
    for index, row in df.iterrows():
        dataset = row['dataset']
        if dataset == datasetName:
            dataFile = str(row['id']) + ".wav"
            freq, data = wavFileToNpy("Data" + PATH_SEPARATOR + dataFile)
            time = float(np.shape(data)[0]) / float(freq)
            speciesId = str(row['speciesId'])
            if speciesId not in speciesToTime:
                speciesToTime[speciesId] = time
            else:
                speciesToTime[speciesId] = speciesToTime[speciesId] + time
    minSpeciesTime = speciesToTime["0"]
    for speciesId in speciesToTime:
        if speciesToTime[speciesId] < minSpeciesTime:
            minSpeciesTime = speciesToTime[speciesId]
    for speciesId in speciesToTime:
        speciesSamplingRatio[speciesId] = minSpeciesTime / speciesToTime[speciesId]
    return speciesSamplingRatio

def shuffleDataAndLabels(x, y):
    rng_state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(rng_state)
    np.random.shuffle(y)

def createDataset(dataFile, sampleLenSeconds, samplesPerMinute):
    df = pd.read_csv(dataFile, sep="\t")
    #print(df.head())
    print("File\tSpecies\tSpeciesId\tSampling Freq (Hz)\tLength (Secs)")
    #Get an even number of samples per species
    speciesSamplingRatio = {}
    speciesSamplingRatio["train"] = computeSpeciesSamplingRatio(df, "train")
    speciesSamplingRatio["validate"] = computeSpeciesSamplingRatio(df, "validate")
    speciesSamplingRatio["test"] = computeSpeciesSamplingRatio(df, "test")

    #process the samples
    samplesPerSpecies = {}
    for index, row in df.iterrows():
        dataFile = str(row['id']) + ".wav"
        species = str(row['species'])
        speciesId = str(row['speciesId'])
        dataset = row['dataset']
        freq, wave = wavFileToNpy("Data/" + dataFile)
        time = float(np.shape(wave)[0]) / float(freq)
        seconds = (str(int(time)) + " Seconds")
        samplingFreq = str(freq) + " Hz"
        print(dataFile + "\t" + species + "\t" + speciesId + "\t" + samplingFreq + "\t" + seconds)
        allSampleStartIndeces, windowsPerSample = sampleWindows(sampleLenSeconds, samplesPerMinute, time, freq, wave)
        xArray, yArray = getDatasetArrays(dataset)
        sampleStartIndeces = np.random.choice(allSampleStartIndeces, int(speciesSamplingRatio[dataset][speciesId] * len(allSampleStartIndeces)), replace=False)
        if dataset not in samplesPerSpecies:
            samplesPerSpecies[dataset] = {}
            samplesPerSpecies[dataset][speciesId] = len(sampleStartIndeces)
        else:
            if speciesId not in samplesPerSpecies[dataset]:
                samplesPerSpecies[dataset][speciesId] = len(sampleStartIndeces)
            else:
                samplesPerSpecies[dataset][speciesId] = samplesPerSpecies[dataset][speciesId] + len(sampleStartIndeces)
        for startIndex in sampleStartIndeces:
            endIndex = startIndex + windowsPerSample
            sample = wave[startIndex:endIndex,]
            #print(np.shape(sample))
            xArray.append(sample)
            yArray.append(speciesId)
        #print(x)
    print("Training samples: " + str(samplesPerSpecies["train"]))
    print("Validation samples: " + str(samplesPerSpecies["validate"]))
    print("Testing samples: " + str(samplesPerSpecies["test"]))

    shuffleDataAndLabels(X_train, y_train)
    shuffleDataAndLabels(X_validate, y_validate)
    shuffleDataAndLabels(X_test, y_test)
    return np.stack(X_train), np.stack(y_train), np.stack(X_validate), np.stack(y_validate), np.stack(X_test), np.stack(y_test)
    #return X_train, y_train, X_validate, y_validate, X_test, y_test

#X_train, y_train, X_validate, y_validate, X_test, y_test = createDataset("data.csv", 10.0, 200)
X_train, y_train, X_validate, y_validate, X_test, y_test = createDataset("data.csv", 12.0, 225)
print("*" * 30)
#model, matrix, acc = trainModel(X_train, y_train, X_validate,y_validate)
numClasses = np.unique(y_train).shape[0]
X_train = addChannelShape(X_train)
X_validate = addChannelShape(X_validate)
y_train = to_categorical(y_train)
y_validate = to_categorical(y_validate)
model = tf.keras.models.Sequential([
    tf.keras.layers.GaussianNoise(0.01),
    tf.keras.layers.Conv1D(128, kernel_size=(31), strides=(6), activation='relu', input_shape=(X_train.shape[1], 1), data_format="channels_last"),
    tf.keras.layers.Conv1D(64, kernel_size=(21), strides=(4), activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv1D(64, kernel_size=(11), strides=(4), activation='relu'),
    tf.keras.layers.Conv1D(64, kernel_size=(11), strides=(4), activation='relu'),
    #tf.keras.layers.MaxPooling1D(pool_size=(2), strides=(2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv1D(64, kernel_size=(11), strides=(4), activation='relu'),
    tf.keras.layers.Conv1D(64, kernel_size=(11), strides=(4), activation='relu'),
    #tf.keras.layers.MaxPooling1D(pool_size=(2), strides=(2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(250, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(numClasses, activation='softmax')
])
#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1.5e-6)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_validate, y_validate), epochs=14)
print(model.summary())
loss, acc = model.evaluate(X_validate, y_validate)#, batch_size=BATCH_SIZE)
print("Loss: " + str(loss))
print("Accuracy: " + str(acc))
y_pred = model.predict(X_validate)
matrix = confusion_matrix(y_validate.argmax(axis=1), y_pred.argmax(axis=1))
print(matrix)
