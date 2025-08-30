import os
from math import tau, pi
import numpy as np
import scipy as sp

from dataclasses import dataclass
from .estimator import *

__all__ = [
    'qCMOS',
    'DMD',
    'SLM',

    'SPADE',
    'PM_SPADE',
    'HG_SPADE',
    'DI',

    'QFI',

    'MetaData',
    'Estimates',
    'LoadEstimates',
    'NewEstimates',

    'Simulator'
]

#################
#    Aliases    #
#################
PM_SPADE_ALIAS = ('spade', 'pm_spade', 'pm')
HG_SPADE_ALIAS = ('hg_spade', 'hg')
DI_ALIAS = ('di',)
ALL_ALIAS = PM_SPADE_ALIAS + HG_SPADE_ALIAS + DI_ALIAS


#################### 
#    Equipments    #
####################
class qCMOS:
    # The camera pixel size is 4.6 um per pixel.
    PIXEL_SIZE = 4.6 #um
    CONVERSION_FACTOR = 0.11
    OFFSET = 200
    QUANTUM_EFFICIENCY_770 = 0.5528 # @770nm


    @classmethod
    def quantum_efficiency(cls, wavelength):
        fx = sp.interpolate.interp1d(np.linspace(250, 1100, 8501), 
                                     np.load(os.path.join(os.path.dirname(__file__), 'quantum_efficiency.npy')))
        return fx(wavelength)
    
    @classmethod
    def convert2photons(cls, img, smoothing=1e-10):
        # If the ADU value is less than the qCMOS offset, means the signal here is zero.
        # 1e-10 for smoothing
        photons = (img - cls.OFFSET) * cls.CONVERSION_FACTOR
        photons = np.clip(photons, smoothing, np.inf)
        return photons


class DMD:
    PIXEL_SIZE = 19.374725804511403 #um

    TRIANGLE_SEQ = np.ravel((np.array([np.arange(-5, 6), np.arange(-5, 6)])).T)[::-1][1:-1]
    TRIANGLE_SEQ = np.concatenate([TRIANGLE_SEQ, TRIANGLE_SEQ[::-1]])

    @classmethod
    def A(cls, px):
        return px * cls.PIXEL_SIZE / 2


class SLM:
    PIXEL_SIZE = 8 #um
    RESOLUTION = (1920, 1080)





######################
#      Data Cls      #
######################
class _Repr:
    def __repr__(self):
        return '\n'.join(f'{attribute}: {value}' for attribute, value in self.__dict__.items())
    
@dataclass
class MetaData(_Repr):
    '''
    Data class to store metadata for measurements.

    Parameters:
        measurement (str): The type of measurement, '(PM/HG)_SPADE' or 'DI'.
        ground_truth (float): The ground truth value of frequency.

        amplitude (float, unit: um): The amplitude value.
        pwm_duty (int, optional): The PWM duty cycle. Defaults to 0. -1 means simulated data with noise.
    '''
    measurement: str
    ground_truth: float
    amplitude: float

    pwm_duty: int = 0

    def __post_init__(self):
        if self.measurement.lower() not in ALL_ALIAS:
            raise ValueError("measurement must be '(PM/HG)_SPADE' or 'DI'")
        else:
            self.measurement = self.measurement.upper()

    def convert2str(self):
        if str(self.pwm_duty).startswith('b='):
            d_str = 'b' + self.pwm_duty[2:]
        else:
            d_str = 'd' + str(self.pwm_duty)
        return f'{self.measurement.lower()}_{round(self.amplitude*2/DMD.PIXEL_SIZE)}px_f{self.ground_truth}_{d_str}'

    def __repr__(self):
        return super().__repr__()


@dataclass
class Estimates(_Repr):
    '''
    A data class to store all data and estimates.

    Parameters:
        metadata (MetaData): MetaData instance.
        
        cropped_data (np.ndarray, unit: ADU): Raw data is a 2D image, we used only 1D data cropped from raw data.
        background (float, unit: photons): Averaged background photons noise per pixel.
        photons (float, unit: photons): Averaged total signal photons used for estimation.

        time_domain (np.ndarray): Time domain signal estimated by MLE localization algorithm.
        estimates (np.ndarray): The frequency estimates. A * sin(2 * pi * f * n + phi)
    '''

    metadata: MetaData

    cropped_data: np.ndarray
    background: float
    photons: float

    time_domain: np.ndarray = None
    estimates: np.ndarray = None

    def td_est(self):
        return td_est(self)

    def freq_est(self):
        return freq_est(self)

    def est(self):
        return freq_est(td_est(self))

    def savez(self, dirname='./'):
        dirname = os.path.expanduser(dirname)
        filename = os.path.join(dirname, self.metadata.convert2str() + '.npz')

        if os.path.exists(filename): 
            raise FileExistsError(f'{filename} already exists')

        self.cropped_data = self.cropped_data.astype(np.uint16)
        np.savez_compressed(filename, **self.__dict__)


    def __repr__(self):
        return super().__repr__()



def LoadEstimates(file) -> Estimates:
    file = os.path.expanduser(file)
    npz = np.load(file, allow_pickle=True)
    dic = {}
    for k in npz.files:
        dic[k] = npz[k]
        if k.lower() == 'cropped_data':
            dic[k] = npz[k].astype(float)
        if k.lower() == 'metadata':
            dic[k] = npz[k].item()
    return Estimates(**dic)



def NewEstimates(raw_path: str, metadata: MetaData) -> Estimates:
    if os.path.exists(raw_path):
        raw = np.load(raw_path)
    else:
        raise FileNotFoundError(f".npy file '{raw_path}' not found")

    raw = raw.astype(float)
    temp = raw[..., :-4, :]
    background = (temp[..., :5,   :5].mean((-1, -2)) + temp[..., -5:,   :5].mean((-1, -2))  +
                  temp[..., :5, -5: ].mean((-1, -2)) + temp[..., -5:, -5: ].mean((-1, -2))) / 4
    background = qCMOS.convert2photons(background).mean()


    if metadata.measurement.lower() in DI_ALIAS:
        lower_bound = int(np.ceil(DI.CENTER + 4*DI.SIGMA / qCMOS.PIXEL_SIZE))
        upper_bound = int(np.ceil(DI.CENTER - (2*metadata.amplitude + 4*DI.SIGMA) / qCMOS.PIXEL_SIZE))  
        cropped = raw[..., upper_bound:lower_bound, DI.X_AXIS]

    elif metadata.measurement.lower() in PM_SPADE_ALIAS:
        cropped = raw[..., (SPADE.POINT_1, SPADE.POINT_2), SPADE.X_AXIS]

    photons = (qCMOS.convert2photons(cropped).mean() - background) * cropped.shape[-1]


    return Estimates(metadata, cropped, background, photons)



######################
#    Measurements    #
######################
class _Share:
    SIGMA = 103 #um
    SAMPLE_LENGTH = 50
    SAMPLING_RATE = 20
    REPEAT = 200

    @classmethod
    def CFI(cls, b, A, f, nu=1, shift=True, delay=0):
        n = np.arange(cls.SAMPLE_LENGTH)
        if type(shift) is not bool:
            raise TypeError('shift must be True or False')
        shift = A if shift else 0

        def _cal(b, f):
            alpha = tau * f * n + delay
            s  = A * np.sin(alpha) + shift
            ds = A*tau*n * np.cos(alpha)
            return (cls.gamma(b, s, nu=nu) * ds**2).sum(-1)

        if np.array(f).ndim != 0:
            return np.array([_cal(b, _f) for _f in f])
        elif np.array(b).ndim != 0:
            return np.array([_cal(_b, f) for _b in b])
        else:
            return _cal(b, f)
        
    @classmethod
    def ApproxCFI(cls, A):
        sum_n = (np.arange(cls.SAMPLE_LENGTH)**2).sum()
        return 2*(A*pi/cls.SIGMA)**2 * sum_n


class SPADE(_Share): # with PM-modes (by default)
    X_AXIS = 89
    POINT_1 = 406
    POINT_2 = 116

    ROI = {'X0': 2128, 'Y0': 720, 'W': 180, 'H': 500}

    @classmethod
    def gamma(cls, b, s, nu=1, smoothing=1e-10):
        b = np.clip(np.atleast_1d(b), smoothing, np.inf) # smoothing
        xi = np.atleast_1d(s) / (2 * cls.SIGMA)
        uk = lambda k: 1 / 2 * (xi + k)**2 * np.exp(-xi**2) + (b / nu)
        duk = lambda k: - 1 / (2 * cls.SIGMA) * (xi + k) * (xi**2 + k * xi - 1) * np.exp(-xi**2)

        return np.array([1 / uk(k) * duk(k)**2 for k in (-1, 1)]).sum(0)
    

class PM_SPADE(SPADE): # alias
    pass


class HG_SPADE(SPADE):
    @classmethod
    def gamma(cls, b, s, nu=1, smoothing=1e-10, maxq=100):
        from scipy.special import factorial
        # s = np.clip(np.atleast_1d(s), smoothing, np.inf) # smoothing
        b = np.clip(np.atleast_1d(b), smoothing, np.inf) # smoothing
        eta = s**2 / (4 * cls.SIGMA**2)

        # if b == 0: # HG-SPADE is vulnerable to noise; even b = 1e-10 can degrade its performance.
        #     gamma_k = lambda k: np.exp(-eta) * eta**(k-1) * (k-eta)**2 / (cls.SIGMA**2 * factorial(k))
        #     return np.array([gamma_k(k) for k in np.arange(maxq+1)]).sum(0)

        uk = lambda k: np.exp(-eta) * eta**k / factorial(k) + (b / nu)
        duk = lambda k: eta**(k - 1) * (k - eta) * (s / (2 * cls.SIGMA**2)) * np.exp(-eta) / factorial(k)

        return np.array([1 / uk(k) * duk(k)**2 for k in np.arange(maxq+1)]).sum(0)
        



class DI(_Share):
    X_AXIS = 86
    CENTER = 113
    
    ROI = {'X0': 1440, 'Y0': 876, 'W': 160, 'H': 228}

    @classmethod
    def gamma(cls, b, s, a=qCMOS.PIXEL_SIZE, regin=np.inf, nu=1, smoothing=1e-10):
        b = np.clip(np.atleast_1d(b), smoothing, np.inf) # smoothing
        s = np.atleast_1d(s)

        from scipy.special import erf
        if regin == np.inf:
            k = np.arange(-10*cls.SIGMA, 10*cls.SIGMA + a, a)
        else:
            k = np.arange(-regin, regin + a, a)

        zp = np.array([(k - _s + 0.5*a) / (cls.SIGMA * (2**0.5)) for _s in np.asarray(s)])
        zn = np.array([(k - _s - 0.5*a) / (cls.SIGMA * (2**0.5)) for _s in np.asarray(s)])
        
        uk = erf(zp)/2 - erf(zn)/2 + (b / nu)
        duk = 1/(cls.SIGMA*(tau**0.5)) * (-np.exp(-zp**2) + np.exp(-zn**2))

        return (1 / uk * duk**2).sum(-1)




#############
#    QFI    #
#############
def QFI(b, A, f, nu=1, delay=0):
    n = np.arange(_Share.SAMPLE_LENGTH)
    
    def _cal(b, f):
        factor = 1 / (1+ 2*(b/nu))
        alpha = tau * f * n + delay
        ds = A*tau*n * np.cos(alpha)
        return nu/_Share.SIGMA**2 * factor * (ds**2).sum(-1) 

    if np.array(f).ndim != 0:
        return np.array([_cal(b, _f) for _f in f])
    elif np.array(b).ndim != 0:
        return np.array([_cal(_b, f) for _b in b])
    else:
        return _cal(b, f)



#####################
#     Simulator     #
#####################
class Simulator:
    def __init__(self, 
                 measurement,
                 amplitude, 
                 ground_truth,

                 waveform: str = 'sign',
                 delay_params = (0, 0),

                 sampling_rate = _Share.SAMPLING_RATE, 
                 repeat = _Share.REPEAT,
                 sample_length = _Share.SAMPLE_LENGTH):
        ''' 
        Parameters:
            measurement (str): The type of measurement, '(PM/HG)_SPADE' or 'DI'.
            ground_truth (float): The ground truth value of frequency.
            amplitude (float, unit: um): The amplitude value.

            waveform (str): 'sign' or 'sin' 
            delay_params (tuple, unit: s): Mean and standard deviation of a Gaussian random delay

            sampling_rate (int): default is 20 Hz
            repeat (int): default is 200
            sample_length (int): default is 50
        '''
        if measurement.lower() in DI_ALIAS:
            measurement = 'DI'
        elif measurement.lower() in PM_SPADE_ALIAS:
            measurement = 'SPADE'
        elif measurement.lower() in HG_SPADE_ALIAS:
            measurement = 'HG'

        self.meta = MetaData(measurement, ground_truth, amplitude)
        self.waveform = waveform.lower()
        self.delay_params = delay_params

        self.fs = sampling_rate
        self.repeat = repeat
        self.N = sample_length


    def _loc(self, n, delay, smooth):
        t = n / self.fs
        fo = self.fs * self.meta.ground_truth

        if self.waveform == 'sign':
            _k = np.sign(np.sin(tau * fo * (t + delay)))
            if _k == 0:
                _k = 1
            return np.clip(self.meta.amplitude * (_k - 1), -np.inf, -smooth)
        
        elif self.waveform == 'sin':
            return np.clip(self.meta.amplitude * (np.sin(tau * fo * (t + delay)) - 1), -np.inf, -smooth)

        else:
            raise ValueError("waveform must be 'sign' or 'sin'")


    def gen(self, noise=0, photons=None, modes=21, smooth=0.01*_Share.SIGMA):
        '''
        Generate simulated data using a statistical histogram method.

        Parameters:
            noise (float, unit: photons): The lambda parameter of the poisson, default is 0.
            photons (int, unit: photons): Number of photons to generate for each sample, default is 400 (DI) or 60 (SPADE).
            modes (int): Number of HG modes used for HG-SPADE. Affects HG-SPADE simulations only. Default is 20.

        Returns:
            np.ndarray: Simulated data array.
        '''
        photons = photons or (400 if self.meta.measurement.lower() in DI_ALIAS else 60)

        if self.meta.measurement.lower() in PM_SPADE_ALIAS:
            _sig = SPADE.SIGMA
            p1 = lambda s: (s-2*_sig)**2*np.exp(-s**2/(4*_sig**2))/(8*_sig**2)
            p2 = lambda s: (s+2*_sig)**2*np.exp(-s**2/(4*_sig**2))/(8*_sig**2)
            def _gen_one(n, delay):
                return np.histogram(np.random.uniform(0, 1, photons), 
                                    [0, 
                                     p1(self._loc(n, delay, smooth)), 
                                     p1(self._loc(n, delay, smooth)) + p2(self._loc(n, delay, smooth))])[0]

        elif self.meta.measurement.lower() in DI_ALIAS:
            _sig = DI.SIGMA / qCMOS.PIXEL_SIZE
            detectors = round((2*self.meta.amplitude + 8*DI.SIGMA) / qCMOS.PIXEL_SIZE)
            def _gen_one(n, delay):
                # Convert length units to camera pixel size to match experimental data.
                loc = (self._loc(n, delay, smooth) + self.meta.amplitude) / qCMOS.PIXEL_SIZE
                outcomes = np.random.normal(detectors/2+loc, _sig, photons)
                return np.histogram(outcomes, bins=detectors, range=(0, detectors))[0]
            
        elif self.meta.measurement.lower() in HG_SPADE_ALIAS:
            _sig = HG_SPADE.SIGMA
            def _gen_one(n, delay):
                _eta = self._loc(n, delay, smooth)**2 / (2*_sig)**2
                outcomes = np.random.poisson(_eta, size=photons)
                return np.histogram(outcomes, bins=np.arange(modes+1))[0]

        data = []
        for _ in range(self.repeat):
            delay = np.random.normal(*self.delay_params)
            for n in range(self.N):
                data.append(_gen_one(n, delay))
        data = np.array(data).astype(float).reshape(self.repeat, self.N, -1)

        if noise == 0:
            self.meta.pwm_duty = 0
        else:
            self.meta.pwm_duty = f'b={np.round(noise / photons, 5)}'

        return Estimates(self.meta, 
                         np.round((data + np.random.poisson(noise, size=data.shape)) / qCMOS.CONVERSION_FACTOR + qCMOS.OFFSET),
                         noise, photons)