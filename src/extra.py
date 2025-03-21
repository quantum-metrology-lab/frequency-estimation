from math import pi, tau
import numpy as np
from scipy.optimize import curve_fit, minimize, brute
from scipy.special import erf
from scipy.stats import norm

from .core import Estimates


__all__ = ['ExEstimates']


def _standardize(signal: np.ndarray):
    '''
    Standardizes the signal using the formula: `(signal - signal.mean()) / signal.std()`.

    Parameters:
        signal (np.ndarray): The input signal to be standardized.

    Returns:
        np.ndarray: The standardized signal with a mean of 0 and unit variance.
    '''
    return (signal - signal.mean()) / signal.std()



class ExEstimates(Estimates):
    def __init__(self, estimates_instance: Estimates):
        '''
        Initializes an instance of `ExEstimates`, extending the `Estimates` class for multi-parameter estimation.

        Parameters:
            estimates_instance (Estimates): `Estimates` instance

        This extended class introduces three additional fields: `extra_estimates_a`, `extra_estimates_f`, and `extra_estimates_phi`, 
        all of which are optional.

        Three methods, `fft`, `mle`, and `lse`, are provided in `ExEstimates`.
        
        Where `mle` follows the method from Kay1993 and jointly estimates amplitude with some approximations.

        This class is mainly used for frequency estimation from time-domain signals. If the time-domain signal is not available, 
        it will be automatically computed by using `Estimates.td_est()` method.
        '''

        super().__init__(**estimates_instance.__dict__)

        self.extra_estimates_a: np.ndarray = None
        self.extra_estimates_f: np.ndarray = None
        self.extra_estimates_phi: np.ndarray = None

        if not isinstance(self.time_domain, np.ndarray):
            self.time_domain = self.td_est().time_domain

    
    def __repr__(self):
        return super().__repr__()



    def fft(self, zero_padding=None, window_type=None):
        N = self.time_domain.shape[-1]
        # Apply windowing function
        if window_type is None:
            window = 1
        elif window_type.lower() == 'hanning':
            window = np.hanning(N)
        elif window_type.lower() == 'hamming':
            window = np.hamming(N) 
        elif window_type.lower() == 'blackman':
            window = np.blackman(N)
        else:
            raise ValueError("window type must be one of 'hanning', 'hamming', or 'blackman'.")

        extra_estimates_f, extra_estimates_phi = [], []
        for sample in self.time_domain:
            # Apply the window to the signal
            windowed_sample = _standardize(sample) * window

            # Zero-padding to increase frequency resolution
            if zero_padding is None:
                N_ = N + 32 if self.metadata.pwm_duty == 0 else N
            else:
                N_ = N + zero_padding

            fft_result = np.fft.fft(windowed_sample, n=N_)
            fft_magnitude = np.abs(fft_result)[:N_ // 2]

            freq = np.fft.fftfreq(N_, 1)[:N_ // 2]
            phase = np.angle(fft_result)[:N_ // 2]

            # Find the peak in the FFT
            peak_index = np.argmax(fft_magnitude[1:]) + 1

            extra_estimates_f.append(freq[peak_index])
            extra_estimates_phi.append(phase[peak_index])

        self.extra_estimates_f = np.array(extra_estimates_f)
        self.extra_estimates_phi = np.array(extra_estimates_phi)
        return self




    def lse(self):
        # Use LSE as frequency estimator. 
        # Note that this LSE is an non-linear LSE.
        N = self.time_domain.shape[-1]
        n = np.arange(N, dtype=np.float64)

        def _s(n, *params):
            f, phi = params
            return (4/pi) * self.metadata.amplitude * np.sin(tau * f * n + phi)

        def _sjac(n, *params):
            f, phi = params
            jac = np.zeros((len(n), 2))
            A = (4/pi) * self.metadata.amplitude
            jac[:, 0] = A * tau * n * np.cos(tau * f * n + phi)
            jac[:, 1] = A * np.cos(tau * f * n + phi)
            return jac
        
        # Use FFT as pre-estimator
        extra_estimates_f, extra_estimates_phi = [], []
        pre_freq_est = self.fft().extra_estimates_f

        for i, sample in enumerate(self.time_domain):
            popt, _= curve_fit(_s, xdata=n, ydata=sample, p0=[pre_freq_est[i], 0], maxfev=1000, jac=_sjac)
            extra_estimates_f.append(popt[0])
            extra_estimates_phi.append(popt[1])

        self.extra_estimates_f = np.array(extra_estimates_f)
        self.extra_estimates_phi = np.array(extra_estimates_phi)
        return self




    def mle(self):
        # Use MLE as frequency estimator. (Kay1993)
        N = self.time_domain.shape[-1]
        n = np.arange(N, dtype=np.float64)

        def I(sample):
            def wrapper(f):
                exr1 = np.sum(sample * np.cos(2*np.pi * f * n))
                exr2 = np.sum(sample * np.sin(2*np.pi * f * n) * n)

                exr3 = np.sum(sample * np.sin(2*np.pi * f * n))
                exr4 = np.sum(sample * np.cos(2*np.pi * f * n) * n)

                return - np.abs((sample * np.exp(-2j*np.pi*f*n)).sum())**2 / N, \
                        2*np.pi/N * (exr1*exr2 - exr3*exr4)
            return wrapper

        # Use FFT as pre-estimator
        extra_estimates_a, extra_estimates_f, extra_estimates_phi = [], [], []
        pre_freq_est = self.fft().extra_estimates_f
        for i, sample in enumerate(self.time_domain):
            result = minimize(I(sample), 
                              pre_freq_est[i],
                              bounds = [(0.05, 0.45)], 
                              tol = 1e-8, 
                              options = {'maxls': 100},
                              jac=True)

            if result.success:
                hat_f = result.x.item()
                hat_A = (2/N) * np.abs((sample * np.exp(-2j*pi * hat_f * n)).sum())
                hat_phi = np.arctan(np.sum(sample * np.cos(tau * hat_f * n) * n) / np.sum(sample * np.sin(tau * hat_f * n) * n))

                extra_estimates_a.append(hat_A)
                extra_estimates_f.append(hat_f)
                extra_estimates_phi.append(hat_phi)

            else:
                raise RuntimeError(f'not converged: {result.message}')

        self.extra_estimates_a = np.array(extra_estimates_a)
        self.extra_estimates_f = np.array(extra_estimates_f)
        self.extra_estimates_phi = np.array(extra_estimates_phi)
        return self


# class MultiParams:
#     N = _Share.SAMPLE_LENGTH
#     sigma = _Share.SIGMA

#     @classmethod
#     def ApproxCFIM(cls, A):
#         n = np.arange(cls.N)
#         mat_1 = [[cls.N, 0, 0],
#                  [0, np.sum(n**2), np.sum(n)],
#                  [0, np.sum(n), cls.N]]
        
#         mat_2 = [[0.5, 0, 0],
#                  [0, 2*A**2*pi**2, pi*A**2],
#                  [0, pi*A**2, A**2*0.5]]

#         return (1/cls.sigma**2) * np.array(mat_1) * np.array(mat_2)

#     @classmethod
#     def ApproxCRB(cls, A):
#         return np.linalg.inv(cls.ApproxCFIM(A))[1, 1]
