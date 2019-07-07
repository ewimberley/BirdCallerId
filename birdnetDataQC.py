#!/usr/bin/python3
from birdnetSignalProcessing import *

import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def wavePlotDataset(dataFile, sampleLenSeconds, numSamples):
    df = pd.read_csv(dataFile, sep="\t")
    print("File\tSpecies\tSpeciesId\tSampling Freq (Hz)\tLength (Secs)")
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
        sampleStartIndeces = np.linspace(0, len(data) - sampleLenSeconds*freq, num=numSamples, dtype=np.int32)
        for startIndex in sampleStartIndeces:
            sample = data[startIndex:int(startIndex+sampleLenSeconds*freq)]
            wavePlot(sample, freq, freq*sampleLenSeconds/10, "QC\\"+str(row['dataset'])+"wave\\" + str(row['id']) + "-" + str(startIndex) + ".png")

def computeSpeciesTime(dataFile, datasetName):
    df = pd.read_csv(dataFile, sep="\t")
    speciesToTime = {}
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
    return speciesToTime

def plotDataset(dataFile, sampleLenSeconds, samplesPerMinute):
    df = pd.read_csv(dataFile, sep="\t")
    #print(df.head())
    X_train = []
    y_train = []
    X_validate = []
    y_validate = []
    X_test = []
    y_test = []
    print("File\tSpecies\tSpeciesId\tSampling Freq (Hz)\tLength (Secs)")
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
        x = np.log10(x+0.000001)
        #x = gaussianNormalization(x)
        #x = maxNormalization(x)
        x = customNormalization(x)
        plotSTFT(f, t, x, "QC\\"+str(row['dataset'])+"\\"+str(row['id'])+".png",ylim_max=20000)
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
        for startIndex in sampleStartIndeces:
            endIndex = startIndex + windowsPerSample
            sample = x[startIndex:endIndex,]
            sampleT = t[startIndex:endIndex]
            plotSTFT(f, sampleT, np.transpose(sample), "QC\\"+str(row['dataset'])+"\\Samples\\"+str(row['id'])+"-"+str(startIndex)+".png",ylim_max=20000)

            #print(sample.shape)
            xArray.append(sample)
            yArray.append(speciesId)
        #print(x)
        #if dataset == "train":

trainingTimes = computeSpeciesTime("data.csv", "train")
print("Training times:")
print(trainingTimes)

validationTimes = computeSpeciesTime("data.csv", "validate")
print("Validation times:")
print(validationTimes)

testTimes = computeSpeciesTime("data.csv", "test")
print("Testing time:")
print(testTimes)

#plotDataset("data.csv", 10.0, 20)
#wavePlotDataset("data.csv", 1.0, 20)

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
