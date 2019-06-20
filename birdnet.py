#!/usr/bin/python3
from birdnet import *

import sys
from scipy.io.wavfile import read
import numpy as np
import pandas as pd
import scipy.signal as signal
import gc

import matplotlib.colors as colors
import matplotlib.pyplot as plt

def wavePlot(inputSignal, samplingFreq, samples, maxTime):
    #t = np.arange(0.0, maxTime, samplingFreq)
    t = np.linspace(0, len(inputSignal)-1, num=samples, dtype=np.int64)
    fig, ax = plt.subplots()
    times = t / samplingFreq
    ax.plot(times, inputSignal[t])
    ax.set(xlabel='time (s)', ylabel='magnitude',
           title='Waveform')
    ax.grid()
    fig.savefig("C:\\Users\\blank\\Documents\\GitHub\\birdCallClassifier\\Data\\test2.png")
    #plt.show()

def plotSTFT(inputSignal, samplingFreq, window='hann', nperseg=256, nfft=256, figsize=(9,5), cmap='magma', ylim_max=None):
    f, t, Zxx = signal.stft(inputSignal, samplingFreq, nfft=nfft, window=window, nperseg=nperseg)
    fig = plt.figure(figsize=figsize)
    ### Different methods can be chosen for normalization: PowerNorm; LogNorm; SymLogNorm.
    ### Reference: https://matplotlib.org/tutorials/colors/colormapnorms.html
    spec = plt.pcolormesh(t, f, np.abs(Zxx),
                          norm=colors.PowerNorm(gamma=1./8.),
                          #norm=colors.LogNorm(vmin=np.abs(Zxx).min(), vmax=np.abs(Zxx).max()),
                          #norm=colors.SymLogNorm(linthresh=0.13, linscale=1, vmin=-1.0, vmax=1.0),
                          cmap=plt.get_cmap(cmap))
    cbar = plt.colorbar(spec)
    ##Plot adjustments
    plt.title('STFT Spectrogram')
    ax = fig.axes[0]
    ax.grid(True)
    ax.set_title('STFT Magnitude')
    if ylim_max:
        ax.set_ylim(0,ylim_max)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    fig.show()
    #plt.show()
    plt.savefig('C:\\Users\\blank\\Documents\\GitHub\\birdCallClassifier\\Data\\test.png')
    return

def STFT(inputSignal, samplingFreq, window='hann', nperseg=256, nfft=256):
    f, t, Zxx = signal.stft(inputSignal, samplingFreq, nfft=nfft, window=window, nperseg=nperseg)
    return f, t, np.abs(Zxx)

def wavFileToNpy(filename):
    wav = read(filename)
    wavNp = np.array(wav[1],dtype=float)
    #np.save(sys.argv[2], wavNp)
    return wav[0], wavNp

def createDataset(dataFile):
    df = pd.read_csv(dataFile, sep="\t")
    #print(df.head())
    X_train = []
    y_train = []
    X_validate = []
    y_validate = []
    X_test = []
    y_test = []
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
        y = row['speciesId']
        f, t, x = STFT(data, freq)
        #print(x)
        #if dataset == "train":

    return X_train, y_train, X_validate, y_validate, X_test, y_test

createDataset("data.csv")
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
