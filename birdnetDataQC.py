#!/usr/bin/python3
from birdnetSignalProcessing import *

import sys
import numpy as np
import pandas as pd

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
        plotSTFT(f, t, x, "QC\\"+str(row['id'])+".png",ylim_max=20000)
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
            plotSTFT(f, sampleT, np.transpose(sample), "QC\\Samples\\"+str(row['id'])+"-"+str(startIndex)+".png",ylim_max=20000)
            exit(0)
            #print(sample.shape)
            xArray.append(sample)
            yArray.append(speciesId)
        #print(x)
        #if dataset == "train":

plotDataset("data.csv", 5.0, 80)
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
