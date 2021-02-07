import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from FractionalOctave.spec import Spec


class Filterbanks:
    r""" Multi-rate filterbanks using fractional-octave filters """
    def __init__(self, sample_rate, spec=Spec(), filter_order=None, dec_stop=80.0, plotting=False):
        r"""
        Construct the filterbanks

        Parameters
        ----------
        sample_rate : float
            Sample rate [Hz]
        spec : Spec
            IEC specifications
        filter_order : Int
            Band filter order
        dec_stop : float
            Minimum stop-band attenuation [dB] for the decimation filter. This value influences the dynamic range for
            spectrogram analysis
        plotting : bool
            Enable plotting
        """
        self._sample_rate = sample_rate

        assert spec.octave_ratio() == 2, 'for base-two system only'
        self._spec = spec
        frac = spec.fraction()

        if filter_order is None:
            if spec.filter_class() == 2:
                filter_order = 3
            else:
                filter_order = 4

        # Allocate bands
        max_w = 0.8
        self._max_band_index = spec.reference_band_index()\
            + int(np.floor(frac * np.log2(max_w * (2 ** (-1/(frac * 2))) * sample_rate
                                          / (spec.reference_frequency() * 2))))

        # Design band filters
        self._filters = list()
        for cnt in range(frac):
            fc, fl, fu = spec.band_frequency(self._max_band_index - (frac - 1 - cnt))
            q_r = fc / (fu - fl)
            q_d = (np.pi / (2 * filter_order)) / np.sin(np.pi / (2 * filter_order)) * q_r
            alpha = (1 + np.sqrt(1 + 4 * (q_d ** 2))) / (2 * q_d)
            f1 = fc / alpha
            f2 = fc * alpha
            self._filters.append(scipy.signal.butter(filter_order, [f1, f2], 'bandpass', output='sos', fs=sample_rate))

        # Design decimation filter
        order, wn = scipy.signal.cheb2ord(max_w / 2, 1 - max_w / 2, 1.0, dec_stop)
        self._dec_filter = scipy.signal.cheby2(order, dec_stop, wn, output='sos')
        ir = np.hstack(([1.0, np.zeros(999)]))
        ir = scipy.signal.sosfilt(self._dec_filter, ir)
        self._dec_filter_ir_length = np.argwhere(ir > (np.max(ir) * 1e-2)).flatten()[-1]
        self._dec_filter_ir_length = (self._dec_filter_ir_length // 2) * 2  # force even
        # print(f'IR len = {self._dec_filter_ir_length}')

        if plotting:
            f_span = np.linspace(0, sample_rate / 2, 1000)
            f_lim = sample_rate * max_w / 2

            H = np.zeros((len(f_span), frac)) * 1j
            for k in range(frac):
                _, H[:, k] = scipy.signal.sosfreqz(self._filters[k], f_span, fs=sample_rate)

            plt.figure()
            plt.plot(f_span, 20 * np.log10(np.abs(H)))
            plt.xlim([0, sample_rate / 2])
            plt.ylim([-90.0, 10.0])
            y_lim = plt.ylim()
            for cnt in range(frac):
                fc, fl, fu = spec.band_frequency(self._max_band_index - (frac - 1 - cnt))
                plt.plot([fl, fl], y_lim, 'k:')
                plt.plot([fu, fu], y_lim, 'k:')
            plt.plot([f_lim, f_lim], y_lim, 'k--')
            plt.grid(True)
            plt.title('Band filters')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Amplitude response [dB]')

            plt.figure()
            plt.plot(f_span, 20 * np.log10(np.abs(scipy.signal.sosfreqz(self._dec_filter, f_span, fs=sample_rate)[1])))
            plt.xlim([0, sample_rate / 2])
            y_lim = [-dec_stop - 10.0, 10.0]
            plt.ylim(y_lim)
            plt.plot(np.ones(2) * f_lim / 2, y_lim, 'k--')
            plt.plot(np.ones(2) * (sample_rate - f_lim) / 2, y_lim, 'k--')
            plt.grid(True)
            plt.title('Decimation filter')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Amplitude response [dB]')

    def check_conformance(self, plotting=False):
        r"""
        Check if band filters conform to IEC standard

        Parameters
        ----------
        plotting : bool
            Enable plotting

        Returns
        -------
        ok : bool
            Band filters conform to IEC standard
        """
        ok = True
        frac = self._spec.fraction()
        fc = None
        for cnt in range(frac):
            fc, *_ = self._spec.band_frequency(self._max_band_index - (frac - 1 - cnt))
            if not self._spec.check_filter_conformance(lambda f: self.band_filter_response(cnt, f, True),
                                                       centre_frequency=fc, is_base_band=True,
                                                       sample_rate=self._sample_rate, plotting=plotting):
                ok = False

        if not self._spec.check_filter_conformance(lambda f: self.band_filter_response(frac - 1, f, False),
                                                   centre_frequency=fc, is_base_band=False,
                                                   sample_rate=self._sample_rate, plotting=plotting):
            ok = False

        return ok

    def band_filter_response(self, fraction_index, freq, is_base_band):
        r"""
        Get amplitude response of a band filter

        Parameters
        ----------
        fraction_index : Int
            Band index (0, 1, ..., fraction - 1) in an octave
        freq : array-like
            Frequencies [Hz] at which the amplitude response is computed
        is_base_band : bool
            = True: compute amplitude response without the decimation filter
            = False: compute amplitude response with the decimation filter

        Returns
        -------
        response : array-like
            Amplitude response [dB]
        """
        if is_base_band:
            H = scipy.signal.sosfreqz(self._filters[fraction_index], freq, fs=self._sample_rate)[1]
            return 20 * np.log10(np.abs(H))
        else:
            H = scipy.signal.sosfreqz(self._filters[fraction_index], np.asarray(freq) * 2, fs=self._sample_rate)[1]
            Hdec = scipy.signal.sosfreqz(self._dec_filter, freq, fs=self._sample_rate)[1]
            return 20 * np.log10(np.abs(H) * np.abs(Hdec))

    def filtering(self, signal, num_octaves=4, ringing_skipped=True):
        r"""
        Do multi-rate filtering on a given signal and get individual band signals

        Parameters
        ----------
        signal : array-like
        num_octaves : Int
            Number of octaves for the spectrogram analysis
        ringing_skipped : bool
            To skip some leading samples in each band to reduce filters' ringing

        Returns
        -------
        bands : list
            List of band data, each element contains {'signal', 'sample_rate', 'start_time', 'fc', 'fl', 'fu'}
        """
        skipped_samples = self._dec_filter_ir_length if ringing_skipped else 0
        frac = self._spec.fraction()
        bands = [None] * (frac * num_octaves)
        fs = self._sample_rate
        start_time = 0.0
        for octave_idx in range(num_octaves):
            for frac_idx in range(frac):
                band = {
                    'sample_rate': fs,
                    'start_time': start_time,
                }
                data_idx = (num_octaves - octave_idx - 1) * frac + frac_idx
                band_idx = self._max_band_index - (octave_idx + 1) * frac + frac_idx + 1
                band['fc'], band['fl'], band['fu'] = self._spec.band_frequency(band_idx)
                band['signal'] = scipy.signal.sosfilt(self._filters[frac_idx], signal)
                bands[data_idx] = band
            signal = scipy.signal.sosfilt(self._dec_filter, signal)
            signal = signal[skipped_samples:-1:2]
            start_time += (skipped_samples / fs)
            fs = fs / 2

        return bands

    def spectrum(self, signal, num_octaves=4, ringing_skipped=True, mode='psd'):
        r"""
        Compute fractional-octave spectrum of a given signal

        Parameters
        ----------
        signal : array-like
        num_octaves : Int
            Number of octaves for the spectrogram analysis
        ringing_skipped : bool
            To skip some leading samples in each band to reduce filters' ringing
        mode : str
            'psd': power spectral density
            'power': total band power

        Returns
        -------
        spectrum : ndarray
            Spectrum
        fc : ndarray
            Bands' centre frequencies
        fl : ndarray
            Bands' lower frequencies
        fu : ndarray
            Bands' upper frequencies
        """
        bands = self.filtering(signal=signal, num_octaves=num_octaves, ringing_skipped=ringing_skipped)
        num_bands = len(bands)
        spectrum = np.zeros(num_bands)
        fc = np.zeros(num_bands)
        fl = np.zeros(num_bands)
        fu = np.zeros(num_bands)
        for idx, band in enumerate(bands):
            fc[idx], fl[idx], fu[idx] = band['fc'], band['fl'], band['fu']
            spectrum[idx] = np.mean(band['signal'] ** 2)

        if mode == 'psd':
            spectrum = spectrum / (fu - fl)
        elif mode != 'power':
            raise ValueError('Invalid spectrogram mode')

        return spectrum, fc, fl, fu


if __name__ == '__main__':
    np.seterr(divide='ignore')
    fb = Filterbanks(44100, plotting=True)
    if fb.check_conformance(plotting=True):
        print('Filterbanks conform to IEC standard')
    else:
        print('Filterbanks do not conform to IEC standard')

    plt.show()
