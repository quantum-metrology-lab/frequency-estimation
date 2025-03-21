from math import pi, tau
import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.special import erf

__all__ = [
    'td_est',
    'freq_est'
]


def td_est(estimates_instance):
    from .core import Estimates, qCMOS, SPADE, DI
    c: Estimates = estimates_instance

    origin_shape = c.cropped_data.shape
    flatten = c.cropped_data.reshape(-1, origin_shape[-1])
    works = flatten.shape[0]
    pixels = origin_shape[-1]

    samples = qCMOS.convert2photons(flatten)
    I0, b = c.photons, c.background

    # SPADE measurement
    if c.metadata.measurement.lower() == 'spade':
        _sig = SPADE.SIGMA
        k = qCMOS.convert2photons(c.cropped_data)[..., 0] / qCMOS.convert2photons(c.cropped_data)[..., 1]
        pre_est = 2 * _sig * (1 - np.sqrt(k)) / (1 + np.sqrt(k))

        def _grad_nll(frame):
            k = np.array([-1, 1])
            def wrapper(theta):
                xi = theta / (2 * _sig)
                uk = I0/2 * (xi + k)**2 * np.exp(-xi**2) + b
                return - np.sum(frame * np.log(uk) - uk - (frame * np.log(frame) - frame)), \
                         np.sum(I0/(2*_sig) * np.exp(-xi**2)*((xi + k) * (xi**2 + k*xi - 1) * (frame/uk - 1)))
            return wrapper

    # DI (Ref. [1])
    elif c.metadata.measurement.lower() == 'di':
        _sig = DI.SIGMA  / qCMOS.PIXEL_SIZE
        pre_est = samples @ np.arange(pixels) / samples.sum(-1)

        def _grad_nll(frame):
            x = np.arange(pixels).astype(float)
            def wrapper(theta):
                z1 = (x - theta + 0.5) / (_sig * (2**0.5))
                z2 = (x - theta - 0.5) / (_sig * (2**0.5))
                DeltaE = erf(z1)/2 - erf(z2)/2
                uk = I0 * DeltaE + b
                return - np.sum(frame * np.log(uk) - uk - (frame * np.log(frame) - frame)), \
                       - I0/(_sig*tau**0.5) * np.sum((-np.exp(-z1**2) + np.exp(-z2**2)) * (frame/uk - 1))
            return wrapper


    # minimize nll
    time_domain = []
    for i in range(works):
        result = minimize(_grad_nll(samples[i]), 
                          x0 = np.ravel(pre_est)[i], 
                          jac = True, 
                          bounds = [(-6*_sig, 6*_sig)], 
                          tol = 1e-8, 
                          options = {'maxls': 100})

        if result.success:
            time_domain.append(result.x.item())
        else:
            raise RuntimeError(f'not converged: {result.message}')


    c.time_domain = np.array(time_domain).reshape(*origin_shape[:-1])

    if c.metadata.measurement.lower() == 'spade':
        c.time_domain = c.time_domain + c.metadata.amplitude

    elif c.metadata.measurement.lower() == 'di':
        c.time_domain = (c.time_domain - pixels/2) * qCMOS.PIXEL_SIZE

    return c





def freq_est(estimates_instance):
    from .core import Estimates
    c: Estimates = estimates_instance
    estimates = []
    a = c.metadata.amplitude

    for sample in c.time_domain:
        # Use FFT as pre-estimator, zero-padding to increase frequency resolution
        # No padding to avoid interference when noisy
        N = len(sample)
        N_ = N + 32 if c.metadata.pwm_duty == 0 else N

        fft = np.abs(np.fft.fft(sample, n=N_))[:N_ // 2]
        freq = np.fft.fftfreq(N_, 1)[:N_ // 2]

        # Find the peak in the FFT
        peak_index = np.argmax(fft[1:]) + 1
        pre_est = freq[peak_index]

        # Use LSE as frequency estimator. (Ref. [2])
        s = lambda n, f: a * np.sin(tau * f * n)
        s_jac = lambda n, f: a * tau * n * np.cos(tau * f * n)
        popt, _ = curve_fit(s, xdata=np.arange(N), ydata=sample, p0=pre_est, maxfev=1000, jac=s_jac)

        estimates.append(popt.item())

    c.estimates = np.array(estimates)
    return c



    

'''
Ref. [1]:
 Carlas S Smith, Nikolai Joseph, Bernd Rieger, et al. 2010. 
 Fast, single-molecule localization that achieves theoretically minimum uncertainty[J/OL]. 
 Nature Methods, April 2010, 7(5): 373-375. https://doi.org/10.1038/nmeth.1449.


Ref. [2]:
 Steven M. Kay. 1993. 
 Fundamentals of Statistical Signal Processing, Volume I: Estimation Theory, 1993[M].
 Pearson.
'''


