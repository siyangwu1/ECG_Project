import scipy
from scipy import signal
import numpy as np


def butter_highpass(data, cutoff, fs, order = 5):
    b, a = signal.butter(N = order, Wn= cutoff, btype='high', analog=False, fs = fs)
    y = signal.filtfilt(b, a, data)
    return y


def down_sampling(data, fs_orig, fs_new):
    return signal.resample_poly(data, up=fs_new, down=fs_orig)


