import numpy as np
import pickle
from tqdm import tqdm


def fft(x):
    maxi = len(x)  # maxi eの配列長
    tmp = np.fft.fft(x)  # tmp eのフーリエ変換
    X = np.zeros(maxi, dtype=complex)
    for i in range(int(maxi / 2)):
        X[i] = tmp[i + int(maxi / 2)]
    for i in range(int(maxi / 2), int(maxi)):
        X[i] = tmp[i - int(maxi / 2)]

    return X


def ifft(X):
    maxi = len(X)
    tmp = np.zeros(maxi, dtype=complex)
    for i in range(int(maxi / 2)):
        tmp[i] = X[i + int(maxi / 2)]
    for i in range(int(maxi / 2), int(maxi)):
        tmp[i] = X[i - int(maxi / 2)]
    x = np.fft.ifft(tmp)

    return x


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def correlation(sq1, sq2):
    if len(sq1) <= len(sq2):
        sq = sq1
        sq_roll = np.roll(sq2, len(sq2) - int(len(sq1) / 2))
    else:
        sq = sq2
        sq_roll = np.roll(sq1, len(sq1) - int(len(sq2) / 2))
    corr = np.zeros(len(sq))
    for i in tqdm(range(len(sq))):
        corr[i] = (len(sq) - np.sum(sq ^ sq_roll[: len(sq)])) / len(sq)
        sq_roll = np.roll(sq_roll, 1)
    return corr


def sampling_signal(signal, n, sampling):
    out = np.zeros(len(signal) // n * sampling, dtype=complex)
    for i in range(len(signal) // n):
        for j, k in enumerate([_ for _ in range(n)][::n // sampling]):
            out[i * sampling + j] = signal[i * n + k]
    return out
