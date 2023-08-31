import matplotlib.pyplot as plt
import numpy as np

# sample freq
fs = 1000

Fstart = 1
Fend = 10

# number of seconds
T = 1

# linearly spaced time vector of T seconds with fs samples per second
x = np.linspace(0, T, fs*T)
c = (Fend-Fstart)/T
y = np.sin(2*np.pi*(c/2 * x**2 + Fstart*x)) + \
    np.random.normal(0, 1, len(x))

# ft = c*x+Fstart

yt = np.sin(2*np.pi*500*x)


N = len(x)
n = np.arange(N)
T_1 = N/fs
freq = n/T_1
# plt.figure()
# plt.plot(x, y)
# plt.title('signal')

# plt.figure()
# plt.plot(x, ft)
# plt.title("frequencies")


fftout = np.fft.fft(y)

plt.figure()
plt.specgram(y, Fs=fs, Fc=0, NFFT=256, scale='default')

plt.show()
