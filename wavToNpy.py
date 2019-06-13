#!/usr/bin/python3
import sys
from scipy.io.wavfile import read
import numpy as np
wav = read(sys.argv[1])
wavNp = np.array(wav[1],dtype=float)
np.save(sys.argv[2], wavNp)
