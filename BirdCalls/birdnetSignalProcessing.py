#!/usr/bin/python3

from scipy.io.wavfile import read
import numpy as np
import pandas as pd
import scipy.signal as signal
import gc

import matplotlib.colors as colors
import matplotlib.pyplot as plt

PATH_SEPARATOR = "/"

def loadFilterNormalize(dataFile):
    freq, data = wavFileToNpy("Data" + PATH_SEPARATOR + dataFile)
    time = float(len(data)) / float(freq)
    f, t, x = STFT(data, freq)
    x = np.log10(x + 0.000001)  # noise filter
    x = customNormalization(x)
    return freq, time, f, t, x

def sampleWindows(sampleLenSeconds, samplesPerMinute, time, windowsPerSec, x):
    numWindows = len(x)
    print("Number of windows: " + str(numWindows))
    #windowsPerSec = int(numWindows / time)  # this is not right?
    print("Windows per second: " + str(windowsPerSec))
    numSamples = int(samplesPerMinute * time / 60.0)
    print("Number of samples: " + str(numSamples))
    windowsPerSample = int(sampleLenSeconds * windowsPerSec)
    print("Windows per sample: " + str(windowsPerSample))
    # print(x.shape)
    # print(x)
    # FIXME cut off first and last 10% of sound?
    sampleStartIndeces = np.linspace(0, numWindows - windowsPerSample, num=numSamples, dtype=np.int32)
    # print(sampleStartIndeces)
    return sampleStartIndeces, windowsPerSample

def customNormalization(x):
    mean = np.mean(x, axis=1)
    std = np.std(x, axis=1)
    x = np.transpose(x)
    x = (x - mean) / std
    #min = abs(np.amin(x))
    #x = x + min
    #max = np.amax(x)
    #x = x / max
    return x

def wavePlot(inputSignal, samplingFreq, samples, fileName):
    t = np.linspace(0, len(inputSignal)-1, num=samples, dtype=np.int64)
    fig, ax = plt.subplots()
    times = t / samplingFreq
    ax.plot(times, inputSignal[t])
    ax.set(xlabel='Time (s)', ylabel='Magnitude',
           title='Waveform')
    ax.grid()
    fig.savefig(fileName)
    plt.close(fig)
    #plt.show()

def plotSTFT(f, t, Zxx, fileName, figsize=(9,5), cmap='magma', ylim_max=None, norm=False):
    fig = plt.figure(figsize=figsize)
    ### Different methods can be chosen for normalization: PowerNorm; LogNorm; SymLogNorm.
    ### Reference: https://matplotlib.org/tutorials/colors/colormapnorms.html
    #spec = plt.pcolormesh(t, f, np.abs(Zxx),
    if norm:
        spec = plt.pcolormesh(t, f, Zxx,
            #norm=colors.LogNorm(vmin=np.abs(Zxx).min(), vmax=np.abs(Zxx).max()),
            norm=colors.PowerNorm(gamma=1./16.),
            cmap=plt.get_cmap(cmap))
    else:
        spec = plt.pcolormesh(t, f, Zxx,
                          #norm=colors.PowerNorm(gamma=1./16.),
                          #norm=colors.LogNorm(vmin=np.abs(Zxx).min(), vmax=np.abs(Zxx).max()),
                          #norm=colors.SymLogNorm(linthresh=0.13, linscale=1, vmin=-1.0, vmax=1.0),
                          cmap=plt.get_cmap(cmap))
    cbar = plt.colorbar(spec)
    ##Plot adjustments
    plt.title('STFT Spectrogram')
    ax = fig.axes[0]
    ax.grid(True)
    ax.set_title('STFT')
    if ylim_max:
        ax.set_ylim(0,ylim_max)
    ax.set_ylabel('Hz')
    ax.set_xlabel('Seconds')
    #fig.show()
    #plt.show()
    plt.savefig(fileName)
    plt.close(fig)
    return

#def STFT(inputSignal, samplingFreq, window='hann', nperseg=256, nfft=1024):
#def STFT(inputSignal, samplingFreq, window='hann', nperseg=256, nfft=512):
#def STFT(inputSignal, samplingFreq, window='hann', nperseg=256, nfft=256):
def STFT(inputSignal, samplingFreq, window='hann', nperseg=256, nfft=320):
    f, t, Zxx = signal.stft(inputSignal, samplingFreq, nfft=nfft, window=window, nperseg=nperseg)
    return f, t, np.abs(Zxx)

def wavFileToNpy(filename):
    wav = read(filename)
    wavNp = np.array(wav[1],dtype=float)
    #np.save(sys.argv[2], wavNp)
    return wav[0], wavNp