"""Feature extraction library for LING 124 at SJSU"""

__author__ = "Hahn Koo (hahn.koo@sjsu.edu)"

import sys
import numpy as np
from scipy import io, fft, signal
import matplotlib.pyplot as plt

def preemphasis(x, alpha=0.97):
	return x - alpha * np.pad(x[:-1], (1, 0), 'constant')

def window(x, frame_size=0.025, frame_shift=0.01, sampling_rate=16000, taper='boxcar'):
	frame_size = int(frame_size * sampling_rate)
	frame_shift = int(frame_shift * sampling_rate)
	n_frames = len(x) // frame_shift + 1
	to_pad = (n_frames - 1) * frame_shift + frame_size - len(x)
	x = np.pad(x, (0, to_pad), 'constant')
	w = getattr(signal.windows, taper)
	frames = np.array([w(frame_size) * x[n*frame_shift : n*frame_shift + frame_size] for n in range(n_frames)])
	return frames

def dft(frames, frame_shift=0.01, sampling_rate=16000):
	t = np.arange(frames.shape[0]) * frame_shift
	f = fft.rfftfreq(frames.shape[1], d=1/sampling_rate)
	m = np.abs(fft.rfft(frames).T)
	return f, t, m

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

def standard(x, sampling_rate=16000, frame_size=0.025, frame_shift=0.01, taper='hamming', alpha=0.97, n_mel_filters=26, n_mfcc=12):
	"""Extract feature vectors consisting of
	(1) energy and 12 MFCCs
	(2) deltas
	(3) delta-deltas
	"""
	frames = window(preemphasis(x, alpha), taper=taper)
	f, t, s = dft(frames)
	mfb = mel_filterbank(26, 0, sampling_rate/2, f)
	ms = np.dot(mfb, s)
	c = mfcc(ms, n_coefficients=n_mfcc)
	e = energy(frames)
	static = np.vstack((e, c))
	d = delta(static)
	return np.concatenate((static, d, delta(d)))

if __name__ == '__main__':
	fs, x = io.wavfile.read(sys.argv[1])
	frames = window(preemphasis(x))
	f, t, s = dft(frames)
	mfb = mel_filterbank(26, 0, fs/2, f)
	ms = np.dot(mfb, s)
	e = energy(frames)
	fig, ax = plt.subplots(4, 1)
	ax[0].pcolormesh(t, f, 10 * np.log10(s))
	ax[1].pcolormesh(t, np.arange(26), 10 * np.log10(ms))
	ax[2].pcolormesh(t, f, 10 * np.log10(cepstral_lifter(s)))
	ax[3].pcolormesh(t, np.arange(26), 10 * np.log10(cepstral_lifter(ms)))
	plt.show()
	print(standard(x, sampling_rate=fs).shape)
