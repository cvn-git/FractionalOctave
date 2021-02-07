import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from FractionalOctave.spec import Spec
from FractionalOctave.filterbanks import Filterbanks


if __name__ == '__main__':
    sample_rate = 48000.0
    rand_signal = np.random.randn(int(round(sample_rate * 20.0))) * np.sqrt(sample_rate / 2)
    sos = scipy.signal.butter(3, 10000, 'high', output='sos', fs=sample_rate)
    filtered_signal = scipy.signal.sosfilt(sos, rand_signal)

    plt.figure()
    n_dft = 4096
    f_span, _, Sxx = scipy.signal.spectrogram(filtered_signal, fs=sample_rate, window='blackman',
                                              nperseg=n_dft, noverlap=n_dft // 2, mode='psd')
    Sxx = np.mean(Sxx, axis=1)
    plt.semilogx(f_span, 10 * np.log10(Sxx))
    plt.grid(True)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power spectral density [dB rel 1/Hz]')
    plt.title('Spectrogram using SciPy')

    np.seterr(divide='ignore')
    fb = Filterbanks(sample_rate=sample_rate, spec=Spec(fraction=6), dec_stop=200.0, plotting=False)
    print(fb.check_conformance(plotting=False))
    plt.figure()
    Sxx, f_span, *_ = fb.spectrum(filtered_signal, num_octaves=11, mode='psd')
    plt.semilogx(f_span, 10 * np.log10(Sxx))
    plt.grid(True)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power spectral density [dB rel 1/Hz]')
    plt.title('Spectrogram using FractionalOctave')

    plt.show()
