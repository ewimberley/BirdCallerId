#!/usr/bin/python3
#from birdnetRNN import *
from birdnetCNN import *
from birdnetSignalProcessing import *

import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def computeSpeciesSamplingRatio(df, datasetName):
    speciesToTime = {}
    speciesSamplingRatio = {}
    for index, row in df.iterrows():
        dataset = row['dataset']
        if dataset == datasetName:
            dataFile = str(row['id']) + ".wav"
            freq, data = wavFileToNpy("Data\\" + dataFile)
            time = len(data) / freq
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
    X_train = []
    y_train = []
    X_validate = []
    y_validate = []
    X_test = []
    y_test = []
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
        freq, data = wavFileToNpy("Data\\" + dataFile)
        samplingFreq = str(freq) + " Hz"
        time = len(data) / freq
        seconds = (str(int(time)) + " Seconds")
        print(dataFile + "\t" + species + "\t" + speciesId + "\t" + samplingFreq + "\t" + seconds)
        f, t, x = STFT(data, freq)
        x = np.log10(x + 0.000001) #noise filter
        x = customNormalization(x)
        x = np.transpose(x)
        numWindows = len(x)
        print("Number of windows: " + str(numWindows))
        windowsPerSec = int(numWindows / time) #this is not right?
        print("Windows per second: " + str(windowsPerSec))
        numSamples = int(samplesPerMinute * time / 60.0)
        print("Number of samples: " + str(numSamples))
        windowsPerSample = int(sampleLenSeconds * windowsPerSec)
        print("Windows per sample: " + str(windowsPerSample))
        #print(x.shape)
        #print(x)
        #FIXME cut off first and last 10% of sound?
        sampleStartIndeces = np.linspace(0, numWindows - windowsPerSample, num=numSamples, dtype=np.int32)
        #print(sampleStartIndeces)
        xArray = X_train
        yArray = y_train
        if dataset == "validate":
            xArray = X_validate
            yArray = y_validate
        elif dataset == "test":
            xArray = X_test
            yArray = y_test
        sampleStartIndeces = np.random.choice(sampleStartIndeces, int(speciesSamplingRatio[dataset][speciesId] * len(sampleStartIndeces)), replace=False)
        if dataset not in samplesPerSpecies:
            samplesPerSpecies[dataset] = {}
            samplesPerSpecies[dataset][speciesId] = len(sampleStartIndeces)
        else:
            if speciesId not in samplesPerSpecies[dataset]:
                samplesPerSpecies[dataset][speciesId] = len(sampleStartIndeces)
            else:
                samplesPerSpecies[dataset][speciesId] = samplesPerSpecies[dataset][speciesId] + len(sampleStartIndeces)
        #normalize input?
        max = np.amax(x)
        x = x / max
        for startIndex in sampleStartIndeces:
            endIndex = startIndex + windowsPerSample
            sample = x[startIndex:endIndex,]
            #print(sample.shape)
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
trainModel(X_train, y_train, X_validate,y_validate)
exit(0)

#Below is for debugging purposes

#freq, data = wavFileToNpy(sys.argv[1])
#freq, data = wavFileToNpy("C:\\Users\\blank\\Documents\\GitHub\\birdCallClassifier\\Data\\163829561.wav")
#freq, data = wavFileToNpy("C:\\Users\\blank\\Documents\\GitHub\\birdCallClassifier\\Data\\164662831.wav")
freq, data = wavFileToNpy("C:\\Users\\blank\\Documents\\GitHub\\birdCallClassifier\\Data\\164553531.wav")

subsampleSecs = 5.0
subsampleEnd = int(freq * subsampleSecs)
data = data[0:subsampleEnd]
time = len(data) / freq
print(str(time) + " Seconds")
wavePlot(data, freq, 2048, time)
plotSTFT(data, freq, nperseg=128, nfft=1024, ylim_max=20000)
