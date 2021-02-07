import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class Signal:
    def __init__(self, seq, form, n=32, SNR=4):
        self.seq = seq
        self.form = form
        self.n = n
        self.SNR = SNR  # [dB]
        self.input = self.seq
        self.wgn_signal = deepcopy(self.input)
        self.signal = {'input': self.seq[int(self.n / 2):: self.n]}

    def display(self, signal):
        sampling_signal = signal[int(self.n / 2):: self.n]
        fig = plt.figure()
        ax = fig.add_subplot()
        line, = ax.plot(sampling_signal.real, sampling_signal.imag, '.')
        ax.xaxis.set_tick_params(direction='in')
        ax.yaxis.set_tick_params(direction='in')
        plt.show()

    def add_wgn(self):
        N0 = 1 / (10 ** (self.SNR / 10))
        noise = np.random.normal(0, np.sqrt(N0 / 2), len(self.input)) + \
                1j * np.random.normal(0, np.sqrt(N0 / 2), len(self.input))
        self.wgn_signal += noise
        self.signal['wgn_signal'] = self.wgn_signal[int(self.n / 2):: self.n]