import numpy as np
import matplotlib.pyplot as plt

def sum_cos(A, F, phi, t):
    return sum( [A[i] * np.cos(2 * np.pi * F[i] * t + phi[i]) for i in range(len(F))] )

Fs = 44100
T = 0.5; N = int(Fs * T)
t = np.arange(N) / Fs
A = [5, 4, 3, 2, -1]
F = [0, 10, 20, 30, 40]
phi = [0, np.pi/2, -np.pi, 0, -np.pi/2]

x = sum_cos(A, F, phi, t)

fig = plt.figure(figsize=(10, 7))
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax1.plot(t, x)
ax1.set_title('(a) waveform')
ax2.stem(F, A, basefmt='', use_line_collection=True)
ax2.set_title('(b) magnitude spectrum')
ax3.stem(F, phi, basefmt='', use_line_collection=True)
ax3.set_ylim(-np.pi - 0.5, np.pi + 0.5)
ax3.set_yticks([-np.round(np.pi, 2), np.round(-np.pi/2, 2), np.round(0, 2), np.round(np.pi/2, 2), np.round(np.pi, 2)])
ax3.set_title('(c) phase spectrum')
plt.subplots_adjust(hspace=0.5)
plt.show()