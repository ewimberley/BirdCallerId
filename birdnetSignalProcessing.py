#!/usr/bin/python3

from scipy.io.wavfile import read
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.special import softmax
import gc

import matplotlib.colors as colors
import matplotlib.pyplot as plt

def wavePlot(inputSignal, samplingFreq, samples, fileName):
    t = np.linspace(0, len(inputSignal)-1, num=samples, dtype=np.int64)
    fig, ax = plt.subplots()
    times = t / samplingFreq
    ax.plot(times, inputSignal[t])
    ax.set(xlabel='time (s)', ylabel='magnitude',
           title='Waveform')
    ax.grid()
    fig.savefig(fileName)
    plt.close(fig)
    #plt.show()

def plotSTFT(f, t, Zxx, fileName, figsize=(9,5), cmap='magma', ylim_max=None):
    fig = plt.figure(figsize=figsize)
    ### Different methods can be chosen for normalization: PowerNorm; LogNorm; SymLogNorm.
    ### Reference: https://matplotlib.org/tutorials/colors/colormapnorms.html
    #spec = plt.pcolormesh(t, f, np.abs(Zxx),
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
    ax.set_title('STFT Magnitude')
    if ylim_max:
        ax.set_ylim(0,ylim_max)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    #fig.show()
    #plt.show()
    plt.savefig(fileName)
    plt.close(fig)
    return

#def STFT(inputSignal, samplingFreq, window='hann', nperseg=256, nfft=1024):
def STFT(inputSignal, samplingFreq, window='hann', nperseg=256, nfft=512):
    f, t, Zxx = signal.stft(inputSignal, samplingFreq, nfft=nfft, window=window, nperseg=nperseg)
    return f, t, np.abs(Zxx)

def wavFileToNpy(filename):
    wav = read(filename)
    wavNp = np.array(wav[1],dtype=float)
    #np.save(sys.argv[2], wavNp)
    return wav[0], wavNp