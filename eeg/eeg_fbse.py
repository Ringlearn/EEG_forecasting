import numpy as np
import pyedflib
import scipy.integrate as integrate
from scipy.special import j0, j1, jn_zeros
import matplotlib.pyplot as plt
from scipy import signal
from pywt import wavedec

def lowpassfilter(data): 
	highcut = 50
	order = 10
	nyquistFreq = 256/2
	high = highcut / nyquistFreq
	sos  = signal.butter(order, high, btype='low', output='sos')
	filtered = signal.sosfilt(sos, data)
	return filtered

def fourier_bessel_coeffs(x, num_coeffs=None):
	if num_coeffs is None:
		num_coeffs = len(x)
	zeroth_roots = jn_zeros(0, num_coeffs)
	t = np.arange(0, len(x))
	a = len(x)
	print("processsing fbse..........")
	coeffs = np.array([2*integrate.trapz(t*x*j0(i*t/a)) / (a**2 * j1(i)**2) for i in zeroth_roots])
	return coeffs

def plot_signal(signal, length, title):
	plt.autoscale()
	plt.plot(signal[:length])
	plt.title(title)
	plt.show()

def plot_magnitude_spectrum(frequencies, coeffs, title,ax):
	magnitude_spect = np.abs(coeffs)
	ax.plot(frequencies, magnitude_spect)
	ax.set_title(title)
	ax.set_xlabel('Fequency(Hz) ')
	ax.set_ylabel('Spectral Power ')

# read eeg signal
f = pyedflib.EdfReader("./chb01_01.edf")
n = f.signals_in_file
print(n)
for i in range(1):
	print("Channel : ", i)
	signal_labels = f.getSignalLabels()
	sigbufs = np.zeros((n, f.getNSamples()[0]))
	sigbufs[i, :] = f.readSignal(i)
	print("total samples of signal ", len(sigbufs[i, :]))

	sample_slice = 256*10  #  sample for 1 minute
	sliced_sample = sigbufs[i, :sample_slice] #  slice 1 minute sample
	print("length of sample ", len(sliced_sample))

	# Apply fbse 
	sliced_sample = lowpassfilter(sliced_sample)
	fbse_coeffs = fourier_bessel_coeffs(sliced_sample)
	fbse_coeffs = fbse_coeffs[range(int(len(sliced_sample)/2))]
	print("length of sample ", len(fbse_coeffs))
	# plot_signal(coeffs,len(coeffs), "FBSE Coefficients")

	# wavelet tranform
	wavelet = wavedec(fbse_coeffs, 'db1', level=4)
	delta, theta, alpha, beta, gamma = wavelet
	print("length of delta", len(delta))
	print("length of theta",len(theta))
	print("length of alpha",len(alpha))
	print("length of beta",len(beta))
	print("length of gamma",len(gamma))


	delta1,delta2, delta3, delta4  = wavedec(delta, 'db1', level=3)
	# exclude coefficient less than 0.4hz
	modified_delta = np.concatenate([delta2, delta3, delta4])
	modified_delta = np.insert(modified_delta, 0, [0]*6, axis=0)

	tpCount     = len(sliced_sample)
	values      = np.arange(int(tpCount/2))
	timePeriod  = tpCount/256
	frequencies = values/timePeriod

	fig, ax = plt.subplots(3, 2)
	fig.suptitle("Channel : "+str(i))
	plt.subplots_adjust(top=0.9)
	fig.tight_layout(h_pad=2)
	ax[0,0].set_title('EEG Signal')
	ax[0,0].set_xlabel('Samples')
	ax[0,0].set_ylabel('Amplitude')
	ax[0,0].plot(sliced_sample)
	plot_magnitude_spectrum(frequencies[:len(modified_delta)], modified_delta, "Delta spectrum ", ax[0,1])
	plot_magnitude_spectrum(frequencies[:len(theta)], theta, "Theta spectrum ", ax[1,0])
	plot_magnitude_spectrum(frequencies[:len(alpha)], alpha, "Alpha spectrum ", ax[1,1])
	plot_magnitude_spectrum(frequencies[:len(beta)], beta, "Beta spectrum ", ax[2,0])
	plot_magnitude_spectrum(frequencies[:len(gamma)], gamma, "Gamma spectrum ", ax[2,1])
	plt.show()