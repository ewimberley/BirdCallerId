#!/usr/bin/python3
#from birdnetRNN import *
from BirdCalls.birdnetCNN import *
from BirdCalls.birdnetSignalProcessing import *

import numpy as np
import pandas as pd

from BirdCalls.birdnetSignalProcessing import sampleWindows, loadFilterNormalize

X_train = []
y_train = []
X_validate = []
y_validate = []
X_test = []
y_test = []

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
            time = float(len(data)) / float(freq)
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
        freq, time, f, t, x = loadFilterNormalize(dataFile)
        seconds = (str(int(time)) + " Seconds")
        samplingFreq = str(freq) + " Hz"
        print(dataFile + "\t" + species + "\t" + speciesId + "\t" + samplingFreq + "\t" + seconds)
        windowsPerSec = int(len(x) / time)
        allSampleStartIndeces, windowsPerSample = sampleWindows(sampleLenSeconds, samplesPerMinute, time, windowsPerSec, x)
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
            sample = x[startIndex:endIndex,]
            #sampleT = t[startIndex:endIndex]
            #plotSTFT(f, sampleT, np.transpose(sample), "QC/" + str(row['dataset']) + "-" + str(row['id']) + "-" + str(startIndex) + ".png", ylim_max=20000, norm=True)
            sample = customNormalization(sample)
            #plotSTFT(f, sampleT, np.transpose(sample), "QC/" + str(row['dataset']) + "-" + str(row['id']) + "-" + str(startIndex) + "-normalized.png", ylim_max=20000, norm=False)
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

X_train, y_train, X_validate, y_validate, X_test, y_test = createDataset("data.csv", 10.0, 180)
#X_train, y_train, X_validate, y_validate, X_test, y_test = createDataset("data.csv", 10.0, 200)
#X_train, y_train, X_validate, y_validate, X_test, y_test = createDataset("data.csv", 12.0, 225)
print("*" * 30)
model, matrix, acc = trainModel(X_train, y_train, X_validate,y_validate)

#Below is for debugging purposes
#freq, data = wavFileToNpy(sys.argv[1])
