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
    #x = np.log10(x + 0.000001)  # noise filter
    #x = customNormalization(x)
    x = np.transpose(x)
    return freq, time, f, t, x

def sample_windows(sample_len_seconds, samples_per_minute, time, windows_per_sec, x, boundary_cutoff=0.01):
    num_windows = len(x)
    print("Number of windows: " + str(num_windows))
    #windowsPerSec = int(num_windows / time)  # this is not right?
    print("Windows per second: " + str(windows_per_sec))
    num_samples = int(samples_per_minute * time / 60.0)
    print("Number of samples: " + str(num_samples))
    windows_per_sample = int(sample_len_seconds * windows_per_sec)
    print("Windows per sample: " + str(windows_per_sample))
    # print(x.shape)
    # print(x)
    # FIXME cut off first and last 10% of sound?
    boundary_cutoff_windows = num_windows * boundary_cutoff
    sampleStartIndeces = np.linspace(boundary_cutoff_windows, num_windows - windows_per_sample - boundary_cutoff_windows, num=num_samples, dtype=np.int32)
    # print(sampleStartIndeces)
    return sampleStartIndeces, windows_per_sample

def customNormalization(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = std + 0.00001
    x = (x - mean) / std
    #min = np.absolute(np.min(x, axis=0))
    #x = x + min
    min = abs(np.amin(x))
    x = x + min
    max = np.amax(x)
    x = x / max
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
            #norm=colors.PowerNorm(gamma=1./16.),
            norm=colors.SymLogNorm(linthresh=0.13, linscale=1, vmin=np.abs(Zxx).min(), vmax=np.abs(Zxx).max()),
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