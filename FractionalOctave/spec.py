import numpy as np
import matplotlib.pyplot as plt


class Spec:
    r""" IEC specifications for fractional-octave filters """
    def __init__(self, fraction=3, filter_class=0, base='two'):
        r"""
        Construct IEC specifications for fractional-octave filters

        Parameters
        ----------
        fraction : Int
            Fraction number for the bandwidth designator
            (e.g. =1 for whole-octave bands, =3 for third-octave bands)
        filter_class : Int
            IEC filter class (0, 1, or 2)
        base : str
            Octave ratio
            'two': for base-two system G = 2
            'ten': for base-ten system G = 10 ^ (3/10)
        """
        self._fraction = fraction
        self._filter_class = filter_class

        if base == 'ten':
            self._G = 10.0 ** (3.0 / 10.0)
        elif base == 'two':
            self._G = 2.0
        else:
            raise ValueError('Invalid base')

        # Omega tables
        pow_val = np.asarray([0, 1/8, 1/4, 3/8, 1/2, 1, 2, 3, 4])
        omega = self._G ** pow_val
        omega_frac = 1 + (((self._G ** (1 / (2 * fraction))) - 1) / ((self._G ** (1 / 2)) - 1)) * (omega - 1)
        self._passband_fmult = omega_frac[0:5]
        self._stopband_fmult = omega_frac[4:]

        if filter_class == 0:
            self._passband_upper_threshold = 0.15
            self._passband_lower_threshold = np.asarray([-0.15, -0.2, -0.4, -1.1, -4.5])
            self._stopband_upper_threshold = np.asarray([-2.3, -18, -42.5, -62, -75])
        elif filter_class == 1:
            self._passband_upper_threshold = 0.3
            self._passband_lower_threshold = np.asarray([-0.3, -0.4, -0.6, -1.3, -5])
            self._stopband_upper_threshold = np.asarray([-2, -17.5, -42, -61, -70])
        elif filter_class == 2:
            self._passband_upper_threshold = 0.5
            self._passband_lower_threshold = np.asarray([-0.5, -0.6, -0.8, -1.6, -5.5])
            self._stopband_upper_threshold = np.asarray([-1.6, -16.5, -41, -55, -60])
        else:
            raise ValueError('Invalid filter class')

    def fraction(self):
        r"""
        Get fraction number for the bandwidth designator

        Returns
        -------
        fraction : Int
            Fraction number for the bandwidth designator
            (e.g. =1 for whole-octave bands, =3 for third-octave bands)
        """
        return self._fraction

    def filter_class(self):
        r"""
        Get IEC filter class

        Returns
        -------
        filter_class : Int
            IEC filter class (0, 1, or 2)
        """
        return self._filter_class

    def octave_ratio(self):
        r"""
        Get octave ratio

        Returns
        -------
        octave_ratio : float
            Octave ratio
        """
        return self._G

    @staticmethod
    def reference_frequency():
        r"""
        Return reference frequency (centre frequency of the "fundamental" band)

        Returns
        -------
        reference_frequency : float
            Reference frequency [Hz]
        """
        return 1e3

    def reference_band_index(self):
        r"""
        Return the band index of the "fundamental" band

        Returns
        -------
        reference_band_index : Int
            Reference band index
        """
        return 10 * self._fraction

    def band_frequency(self, band_index):
        r"""
        Return band frequency for a given band index

        Parameters
        ----------
        band_index : Int
            Band index

        Returns
        -------
        fc : float
            Band's centre frequency
        fl : float
            Band's lower frequency
        fu : float
            Band's upper frequency
        """
        band_mul = self._G ** (1 / self._fraction)
        f_mul = self._passband_fmult[-1]
        fc = self.reference_frequency() * (band_mul ** (band_index - self.reference_band_index()))
        fl = fc / f_mul
        fu = fc * f_mul
        return fc, fl, fu

    def check_filter_conformance(self, filter_response, centre_frequency, is_base_band, sample_rate, plotting=False):
        r"""
        Check if a band filter conforms to IEC standard

        Parameters
        ----------
        filter_response : Callable
            Function f(freq) to compute band filter response [dB]
        centre_frequency : float
            Band's centre frequency [Hz]
        is_base_band : bool
            = True: compute amplitude response without the decimation filter
            = False: compute amplitude response with the decimation filter
        sample_rate : float
            Sample rate [Hz]
        plotting : bool
            Enable plotting

        Returns
        -------
        ok : bool
            Band filter conforms to IEC standard

        """
        N = 10000
        err_cnt = 0

        if not is_base_band:
            centre_frequency = centre_frequency / 2

        f_ref = np.hstack(([0.0], centre_frequency / np.flip(self._stopband_fmult)))
        f1 = np.linspace(f_ref[0], f_ref[-1], N)
        Href = np.flip(self._stopband_upper_threshold)
        Href = np.hstack(([Href[0]], Href))
        up1 = np.interp(f1, f_ref, Href)
        H1 = filter_response(f1)
        err_cnt += len(np.where(H1 > up1)[0])

        f_ref = np.hstack((centre_frequency / np.flip(self._passband_fmult[1:]),
                           centre_frequency * self._passband_fmult))
        f2 = np.linspace(f_ref[0], f_ref[-1], N)
        up2 = np.ones(N) * self._passband_upper_threshold
        Href = self._passband_lower_threshold
        Href = np.hstack((np.flip(Href[1:]), Href))
        lo2 = np.interp(f2, f_ref, Href)
        H2 = filter_response(f2)
        err_cnt += len(np.where(H2 > up2)[0])
        err_cnt += len(np.where(H2 < lo2)[0])

        f_ref = centre_frequency * self._stopband_fmult
        Href = self._stopband_upper_threshold
        if f_ref[-1] < (sample_rate / 2):
            f_ref = np.hstack((f_ref, [sample_rate / 2]))
            Href = np.hstack((Href, Href[-1]))
        f3 = np.linspace(f_ref[0], sample_rate / 2, N)
        up3 = np.interp(f3, f_ref, Href)
        H3 = filter_response(f3)
        err_cnt += len(np.where(H3 > up3)[0])

        if plotting:
            # Plot upper bound
            cmd = [[np.hstack((f1, f2, f3)), np.hstack((up1, up2, up3)), 'k:']]
            # Plot lower bound
            cmd.append([np.hstack(([f2[0]], f2, [f2[-1]])), np.hstack(([-1e3], lo2, [-1e3])), 'k:'])
            # Plot H
            cmd.append([np.hstack((f1, f2, f3)), np.hstack((H1, H2, H3)), 'b'])

            plt.figure()
            plt.subplot(2, 1, 1)
            for c in cmd:
                plt.plot(*c)
            delta = (f2[-1] - f2[0]) / 3
            plt.xlim([f2[0] - delta, f2[-1] + delta])
            plt.ylim([lo2[0] - 2, up2[0] + 2])
            plt.grid(True)
            if is_base_band:
                plt.title('Band filter')
            else:
                plt.title('Band filter at the upper rate')

            plt.subplot(2, 1, 2)
            for c in cmd:
                plt.plot(*c)
            plt.xlim([0.0, sample_rate / 2])
            plt.ylim([-80.0, 10.0])
            plt.grid(True)
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Amplitude response [dB]')

        return err_cnt == 0


if __name__ == '__main__':
    pass
