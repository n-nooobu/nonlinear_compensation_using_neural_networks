from pyopt.modulate import prbs, Modulate
from pyopt.util import fft, ifft, save_pickle, load_pickle
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm


class Signal:
    def __init__(self, seq, form='RZ16QAM', n=32, baudrate=28, PdBm=0.0, Ledfa=100, stepedfa=30,
                 gamma=1.4, D=16, Alpha=0.16, NF=4):
        self.seq = seq
        self.form = form
        self.n = n
        self.baudrate = baudrate
        self.PdBm = PdBm
        self.Ledfa = Ledfa
        self.stepedfa = stepedfa
        self.gamma = gamma
        self.D = D
        self.Alpha = Alpha
        self.NF = NF

        """固定パラメータ"""
        self.bitratelist = {'NRZQPSK': self.baudrate * 2,
                            'RZQPSK': self.baudrate * 2,
                            'NRZ16QAM': self.baudrate * 4,
                            'RZ16QAM': self.baudrate * 4}
        self.bitrate = self.bitratelist[self.form]  # ビットレート[Gbps]
        self.B = self.baudrate * 2 * 10 ** 9  # 帯域幅[bit/s]
        self.c = 3 * 10 ** 5  # ファイバ中光速[km/sec]
        self.Lambda_c = 1550  # キャリア波長[nm]
        self.f_c = self.c * 10 ** 3 / (self.Lambda_c * 10 ** -9)  # キャリア周波数[Hz]
        self.Omega_c = 2 * np.pi * self.c * 10 ** 3 / self.Lambda_c  # キャリア光角周波数[G rad/s]
        self.nc = 1.46  # コア屈折率
        self.a = 5 * 10 ** -6  # コア半径[m]
        self.As = np.pi * self.a ** 2  # コア面積[m^2]
        self.Xi = 1.5  # コア面積補正
        self.B2 = (-self.D * self.Lambda_c ** 2 * 10 ** -6) / (2 * np.pi * self.c)  # 2次分散[ns^2/km]
        self.G = 10 ** (self.Alpha * self.Ledfa / 10)  # EDFA利得
        self.h = 6.626 * 10 ** -34  # h[j*s] プランク定数
        self.n2 = self.gamma * 10 ** -3 * self.Lambda_c * 10 ** -9 * self.nc * self.Xi * self.As / (240 * np.pi ** 2)
        self.nsp = 0.5 * 10 ** (self.NF / 10)

        """初期信号"""
        self.PW = 10 ** (self.PdBm / 10 - 3)  # 光電力[W]
        self.Phi_p = np.sqrt((120 * np.pi / self.nc) * self.PW / (self.Xi * self.As))  # 光電界強度[V/m]
        self.signal = {'Lnow': 0, 'x_Lnow': self.Phi_p * self.seq}
        self.signal['x_0'] = self.signal['x_Lnow']

        self.symbol, self.index, self.inverse, self.counts = np.unique(self.seq[self.n//2::self.n], return_index=True, return_inverse=True, return_counts=True)

        """軸生成"""
        self.length = len(self.signal['x_Lnow'])
        self.Frequency = self._frequency()  # 周波数軸[GHz]
        self.Lambda = self._wavelength()  # 波長軸[nm]
        self.Omega = self._omegafrequency() * 10 ** -9  # 光角周波数軸[G rad]

    def _timeslot(self):  # Ts[ps/bit] bitあたりの時間長
        Ts = 1 / (self.bitrate * 10 ** 9) * 10 ** 12
        return Ts

    def _frequency(self):  # Lambda_c[nm] 中心波長 Speed[Gb/s]
        Ts = self._timeslot()  # Ts[ps] 1タイムスロット(n点)の時間
        dt = Ts / self.n  # dt[ps] 1点の時間
        fc = 3 * 10 ** 8 / (self.Lambda_c * 10 ** -9) * 10 ** -9  # fc[GHz]
        df = 1 / (dt * (10 ** -12) * self.length) * 10 ** -9  # df[GHz]
        fwidth = 1 / (2 * dt) * 10 ** 3
        Frequency = np.zeros(self.length, dtype=float)
        for i in range(self.length):
            Frequency[i] = fc + (fwidth - df * i)  # f[GHz]
        return Frequency

    def _wavelength(self):  # Lambda_c[nm] 中心波長 Speed[Gb/s]
        fr = self._frequency()  # f[GHz] 周波数軸
        Lambda = 3 * 10 ** 8 / (fr * 10 ** 9) * 10 ** 9  # Lambda[nm] 波長軸
        return Lambda

    def _omegafrequency(self):  # Lambda_c[nm] 中心波長 Speed[Gb/s]
        fr = self._frequency()  # f[GHz] 周波数軸
        Omega = 2 * np.pi * fr * 10 ** 9  # Omega[rad] 角周波数軸
        return Omega

    def phase_rotation_theo(self, L):
        Alpha2 = self.Alpha * 10 ** -4 * np.log(10)
        pr_theo = 2 * np.pi * self.n2 / (self.Lambda_c * 10 ** -9) \
                    * (1 - np.exp(-Alpha2 * self.Ledfa * 10 ** 3)) / Alpha2 \
                    * self.Phi_p ** 2 * int(L / self.Ledfa) * 180/np.pi
        return pr_theo

    def phase_rotation_simu(self, L, signal):
        input_signal = self.signal['x_0'][int(self.n / 2):: self.n]
        tmp = np.zeros((2, 2), dtype=complex)
        for i in range(len(input_signal)):
            if input_signal[i].real > 0 and input_signal[i].imag > 0:
                tmp = np.append(tmp, np.array([input_signal[i], signal[i]]).reshape(1, 2), axis=0)
        tmp = tmp[2::]
        r_in = np.angle(tmp[:, 0]) * 180 / np.pi
        r = np.angle(tmp[:, 1]) * 180 / np.pi
        pr_simu = np.mean(-(r - r_in))
        return pr_simu

    def cal_evm(self, L, signal):
        input_signal = self.signal['x_0'][int(self.n / 2):: self.n]
        tmp = 0
        for i in range(len(signal)):
            tmp += abs(signal[i] - input_signal[i]) ** 2 / abs(input_signal[i]) ** 2
        evm = np.sqrt(tmp / len(signal)) * 100
        return evm

    def cal_evm_pr(self, L, signal):
        input_signal = self.signal['x_0'][int(self.n / 2):: self.n]
        t_sum = 0
        for i in range(len(self.index)):
            t_sum += abs(input_signal[self.index[i]])
        mean = t_sum / len(self.index)

        pr_theo = self.phase_rotation_theo(L)
        signal = signal * np.exp(1j * pr_theo * np.pi / 180)
        tmp = 0
        for i in range(len(signal)):
            tmp += abs(signal[i] - input_signal[i]) ** 2 / abs(input_signal[i]) ** 2
        evm_pr = np.sqrt(tmp / len(signal)) * 100
        return evm_pr

    def cal_evm_min(self, L, signal):
        input_signal = self.signal['x_0'][int(self.n / 2):: self.n]
        evm_min = 200
        i_min = 0
        for i in range(360):
            signal = signal * np.exp(1j * i * np.pi/180)
            tmp = 0
            for j in range(len(signal)):
                tmp += abs(signal[j] - input_signal[j]) ** 2 / abs(input_signal[j]) ** 2
            evm = np.sqrt(tmp / len(signal)) * 100
            if evm < evm_min:
                evm_min = evm
                i_min = i
        return evm_min, i_min

    def cal_ber(self, L, signal):
        evm = self.cal_evm(L, signal)
        M = {'NRZQPSK': 4,
             'RZQPSK': 4,
             'NRZ16QAM': 16,
             'RZ16QAM': 16}
        ber = (1 - M[self.form] ** (-1 / 2)) / (1 / 2 * np.log2(M[self.form])) \
            * special.erfc(np.sqrt(3 / 2 / (M[self.form] - 1) / (evm / 100) ** 2))
        return ber

    def cal_qfac(self, L, signal):
        ber = self.cal_ber(L, signal)
        q = 20 * np.log10(np.sqrt(2) * special.erfcinv(2 * ber))
        return q

    def cal_snr(self, L, signal):
        evm = self.cal_evm(L, signal)
        snr = 10 * np.log10(1 / (evm / 100) ** 2)
        return snr

    def transmission(self, Lmax=1000, ase=True):
        for _ in tqdm(range(int((Lmax - self.signal['Lnow']) / self.Ledfa))):
            self.transmission_1span(ase)

    def transmission_1span(self, ase=True):
        tmp = self.signal['x_Lnow']
        tmp = self._add_nonlinear_distortion(tmp)  # 線形歪,非線形歪,減衰を同時に加える
        tmp *= 10 ** (self.Alpha * self.Ledfa / 10)  # EDFAによる増幅を行う
        if ase:
            tmp = self._add_ase_noise(tmp)  # 増幅と同時にASE雑音を加える
        self.signal['x_Lnow'] = tmp
        self.signal['Lnow'] += self.Ledfa
        if self.signal['Lnow'] % 500 == 0:
            self.signal['x_'+str(self.signal['Lnow'])] = tmp

    def linear_compensation(self, L, x):
        S = fft(x)
        S /= self._H(L)
        x = ifft(S)
        return x

    def _H(self, L):  # H ファイバの伝達関数
        out = np.exp(-1j * self.B2 * L / 2 * (self.Omega - self.Omega_c) ** 2)
        return out

    def _sf(self, x, y=None):
        if y is None:
            out = x * -1j * 2 * np.pi * self.n2 / (2 * self.Lambda * 10 ** -9) * abs(x ** 2)
        else:
            out = x * -1j * 8 / 9 * 2 * np.pi * self.n2 \
                  / (2 * self.Lambda * 10 ** -9) * (abs(x) ** 2 + abs(y) ** 2)  # /2必要？
        return out

    def _runge_kutta(self, L, x, y=None):
        if y is None:
            k1 = self._sf(x) * L
            k2 = self._sf(x + k1 / 2) * L
            k3 = self._sf(x + k2 / 2) * L
            k4 = self._sf(x + k3) * L
            xx = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            x = x * np.exp(-1j * L / 2 * 2 * np.pi * self.n2
                           / (self.Lambda * 10 ** -9) * (abs(x ** 2) + abs(xx ** 2)))
        else:
            k1 = self._sf(x, y) * L
            k2 = self._sf(x + k1 / 2, y) * L
            k3 = self._sf(x + k2 / 2, y) * L
            k4 = self._sf(x + k3, y) * L
            xx = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            x = x * np.exp(-1j * L / 2 * 2 * np.pi * self.n2
                           / (self.Lambda * 10 ** -9) * (abs(x) ** 2 + abs(xx) ** 2))
        return x

    def _add_ase_noise(self, x):
        r = 2 * np.pi * np.random.rand(len(x))
        s = (self.G - 1) * self.nsp * self.h * self.f_c * self.B
        ase_noise = np.sqrt((120 * np.pi / self.nc) * s / (self.Xi * self.As)) * np.exp(1j * r)
        x += ase_noise
        return x

    def _add_nonlinear_distortion(self, x, y=None):
        dL = self.Ledfa / self.stepedfa  # dL[km] 区間一つの長さ
        dH = self._H(dL / 2)  # dH 区間一つの半分における伝達関数
        ddL = dL * 10 ** 3  # ddL[m]

        for i in range(self.stepedfa):
            """1区間の前半の周波数領域での線形歪を計算"""
            X = fft(x)
            X = X * dH
            x = ifft(X)

            """伝搬損失の計算"""
            x = x * 10 ** (-self.Alpha * dL / 2 / 10)  # Alpha = 0.20dB/km dL = 100km のとき1/100

            """1区間の実時間領域での非線形効果による位相推移をルンゲ・クッタ法を用いて計算"""
            x = self._runge_kutta(ddL, x, y)

            """1区間の後半の周波数領域での線形歪を計算"""
            X = fft(x)
            X = X * dH
            x = ifft(X)

            """伝搬損失の計算"""
            x = x * 10 ** (-self.Alpha * dL / 2 / 10)  # Alpha = 0.20dB/km dL = 100km のとき1/100
        return x


def display_constellation(signal, dtype='complex'):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()
    if dtype == 'complex':
        line, = ax.plot(signal.real, signal.imag, '.')
    elif dtype == 'array':
        line, = ax.plot(signal[:, 0], signal[:, 1], '.')
    # ax.legend()
    # ax.set_xlim((-150000, 150000))
    # ax.set_ylim((-150000, 150000))
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_tick_params(direction='in')
    plt.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False)
    plt.tick_params(bottom=False,
                    left=False,
                    right=False,
                    top=False)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
    plt.show()


def display_constellation_color(signal, seq, count=False):
    symbol, inverse, counts = np.unique(seq, return_inverse=True, return_counts=True)

    if count:
        fig = plt.figure(figsize=(6.75, 6))
    else:
        fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()
    for i in range(len(symbol)):
        line, = ax.plot(signal[seq == symbol[i]].real, signal[seq == symbol[i]].imag, '.', label=str(counts[i]))
    if count:
        ax.legend(fontsize=14, markerscale=6)
        ax.set_xlim((-max(signal), max(signal) + 30000))
    else:
        ax.set_xlim((-max(signal), max(signal)))
    ax.set_ylim((-max(signal), max(signal)))
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_tick_params(direction='in')
    plt.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False)
    plt.tick_params(bottom=False,
                    left=False,
                    right=False,
                    top=False)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
    plt.show()


if __name__ == '__main__':

    bitsq = prbs(N=17, itr=0)
    mdl = Modulate('RZ16QAM')
    sq = mdl.transform(bitsq)
    
    sgnl = Signal(seq=sq, form='RZ16QAM', baudrate=28, PdBm=5, gamma=1.4, D=16, Alpha=0.16, NF=4)
    sgnl.transmission(Lmax=1200, ase=True)
    lc = sgnl.linear_compensation(1200, sgnl.signal['x_Lnow'])
    EVM = sgnl.cal_evm(1200, lc[16::32])
    print(EVM)

    # save_pickle(sgnl, '../data/input/N17.pickle')

    """random = np.random.randint(0, 2, 1000)
    mdl = Modulate('RZ16QAM')
    sq = mdl.transform(random)

    sgnl = Signal(seq=sq, form='RZ16QAM')
    sgnl.transmission(Lmax=500, ase=True)

    save_pickle(sgnl, '../data/input/random00000.pickle')"""
