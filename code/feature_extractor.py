"""Feature extraction library for LING 124 at SJSU"""

__author__ = "Hahn Koo (hahn.koo@sjsu.edu)"

import sys
import numpy as np
from scipy import io, fft, signal
import matplotlib.pyplot as plt

def preemphasis(x, alpha=0.97):
	return x - alpha * np.pad(x[:-1], (1, 0), 'constant')

def energy(frames):
	return np.array([np.dot(frame, frame) for frame in frames])

def hz2mel(hz):
	return 1125 * np.log(1 + hz/700)

def mel2hz(mel):
	return 700 * (np.exp(mel/1125) - 1)

def mel_filterbank(n_filters, begin_hz, end_hz, freqs_hz):
	boundaries = mel2hz(np.linspace(hz2mel(begin_hz), hz2mel(end_hz), n_filters + 2))
	b = boundaries[:-2]
	m = boundaries[1:-1]
	e = boundaries[2:]
	within = np.array([f >= b for f in freqs_hz]) * np.array([f <= e for f in freqs_hz])
	left = np.array([f <= m for f in freqs_hz]) * within
	right = np.array([f > m for f in freqs_hz]) * within
	left_weight = np.array([(f - b) / (m - b) for f in freqs_hz])
	right_weight = np.array([(e - f) / (e - m) for f in freqs_hz])
	return (left_weight * left + right_weight * right).T

def cepstral_lifter(magnitude_spectrum, n_coefficients=12):
	c = fft.dct(np.log(magnitude_spectrum), norm='ortho', axis=0)
	if magnitude_spectrum.ndim == 1:
		c = np.array([c]).T
	cs = np.pad(c[:n_coefficients, :], ((0, c.shape[0] - n_coefficients), (0, 0)))
	liftered = np.exp(fft.idct(cs, norm='ortho', axis=0))
	if magnitude_spectrum.ndim == 1:
		liftered = liftered[:, 0]
	return liftered

def mfcc(mel_filtered_spectrum, n_coefficients=12):
	c = fft.dct(np.log(mel_filtered_spectrum), norm='ortho', axis=0)
	return c[:n_coefficients]

def delta(x):
	y = np.pad(x, ((0, 0), (1, 1)), 'edge')
	return (y[:, 2:] - y[:, :-2]) / 2

def mel_spectrogram(x, sampling_rate, frame_size=0.025, frame_shift=0.01, window='hann', boundary=None, padded=False, alpha=0.97, n_mel_filters=40):
	f_hz, t, z = signal.stft(x, fs=sampling_rate, nperseg=int(sampling_rate*frame_size), noverlap=int(sampling_rate*(frame_size-frame_shift)), window=window, boundary=boundary, padded=padded)
	s = np.abs(z)
	mfb = mel_filterbank(n_mel_filters, f_hz[0], f_hz[-1], f_hz)
	ms = np.dot(mfb, s)
	return ms

def standard39(x, sampling_rate, frame_size=0.025, frame_shift=0.01, window='hann', boundary=None, padded=False, alpha=0.97, n_mel_filters=26, use_energy=False):
	"""Extract feature vectors consisting of
	(1) 13 MFCCs or energy and 12 MFCCs
	(2) deltas
	(3) delta-deltas
	"""
	n_coefficients = 13
	if use_energy: n_coefficients = 12
	f_hz, t, z = signal.stft(x, fs=sampling_rate, nperseg=int(sampling_rate*frame_size), noverlap=int(sampling_rate*(frame_size-frame_shift)), window=window, boundary=boundary, padded=padded)
	s = np.abs(z)
	mfb = mel_filterbank(n_mel_filters, f_hz[0], f_hz[-1], f_hz)
	ms = np.dot(mfb, s)
	static = mfcc(ms, n_coefficients=n_coefficients)
	if use_energy: static = np.vstack((energy(frames), static))
	d = delta(static)
	return np.concatenate((static, d, delta(d)))

if __name__ == '__main__':
	fs, x = io.wavfile.read(sys.argv[1])
	ms = mel_spectrogram(x, fs)
	print('# mel spectrogram computed. shape =', ms.shape)
	mfccs = standard39(x, fs)
	print('# mfccs computed. shape =', mfccs.shape)
	fig, ax = plt.subplots(3, 1)
	ax[0].plot(x); ax[0].set_xlim(xmin=0); ax[0].set_title('waveform')
	ax[1].pcolormesh(10*np.log10(ms)); ax[1].set_title('mel spectrogram')
	ax[2].pcolormesh(10*np.log10(cepstral_lifter(ms))); ax[2].set_title('low-pass liftered mel spectrogram')
	plt.subplots_adjust(hspace=0.5)
	plt.show()
