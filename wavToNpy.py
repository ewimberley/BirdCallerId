#!/usr/bin/python3
import sys
from scipy.io.wavfile import read
import numpy as np
import scipy.signal as signal
import gc

import matplotlib.colors as colors
import matplotlib.pyplot as plt

def calcSTFT_norm(inputSignal, samplingFreq, window='hann', nperseg=256, nfft=256, figsize=(9,5), cmap='magma', ylim_max=None):
    '''Calculates the STFT for a time series:
        inputSignal: numpy array for the signal (it also works for Pandas.Series);
        samplingFreq: the sampling frequency;
        window : str or tuple or array_like, optional
            Desired window to use. If `window` is a string or tuple, it is
            passed to `get_window` to generate the window values, which are
            DFT-even by default. See `get_window` for a list of windows and
            required parameters. If `window` is array_like it will be used
            directly as the window and its length must be nperseg. Defaults
            to a Hann window.
        nperseg : int, optional
            Length of each segment. Defaults to 256.
        figsize: the plot size, set as (6,3) by default;
        cmap: the color map, set as the divergence Red-Yellow-Green by default;
        ylim_max: the max frequency to be shown. By default it's the half sampling frequency.'''
    ##Calculating STFT
    print(inputSignal)
    f, t, Zxx = signal.stft(inputSignal, samplingFreq, nfft=nfft, window=window, nperseg=nperseg)
    print(f)
    print(t)
    print(Zxx)
    ##Plotting STFT
    fig = plt.figure(figsize=figsize)
    ### Different methods can be chosen for normalization: PowerNorm; LogNorm; SymLogNorm.
    ### Reference: https://matplotlib.org/tutorials/colors/colormapnorms.html
    spec = plt.pcolormesh(t, f, np.abs(Zxx),
                          norm=colors.PowerNorm(gamma=1./8.),
                          #norm=colors.LogNorm(vmin=np.abs(Zxx).min(), vmax=np.abs(Zxx).max()),
                          #norm=colors.SymLogNorm(linthresh=0.13, linscale=1,
                          #                       vmin=-1.0, vmax=1.0),
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

def wavFileToNpy(filename):
    wav = read(filename)
    wavNp = np.array(wav[1],dtype=float)
    print(str(wav[0]) + " Hz")
    #np.save(sys.argv[2], wavNp)
    return wav[0], wavNp

#wavFileToNpy(sys.argv[1])
freq, data = wavFileToNpy("C:\\Users\\blank\\Documents\\GitHub\\birdCallClassifier\\Data\\163829561.wav")
time = len(data) / freq
print(str(time) + " Seconds")
#calcSTFT_norm(data, 5e6, nperseg=1048576, ylim_max=300000)
calcSTFT_norm(data, freq, nperseg=128, nfft=1024, ylim_max=30000)
