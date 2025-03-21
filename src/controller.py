import os
import numpy as np
from datetime import datetime, timedelta, timezone
from time import sleep

import warnings
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings('ignore', category=CryptographyDeprecationWarning)
import paramiko

from .api import *

__all__ =[
    '__version__',

    'Dcamapi',
    'Dcam',
    'ALP4',

    'EasyDcam',
    'EasyALP4',

    'RaspiLED'
]

__version__ = 0.1


class EasyDcam(Dcam):
    def __init__(self, iDevice=0):
        super().__init__(iDevice)
        self.ez_isopen = False


    def __enter__(self):
        Dcamapi.init()
        self.dev_open()
        self.prop_setvalue(DCAM_IDPROP.SENSORCOOLER, DCAMPROP.SENSORCOOLER.MAX)
        print(f'qCMOS found, current sensor temperature is {self.ez_temperature()}.')
        self.ez_isopen = True
        return self


    def __exit__(self, *args):
        if self.ez_isopen:
            self.cap_stop()
            self.buf_release()
            self.dev_close()
            Dcamapi.uninit()
            self.ez_isopen = False
            print('EasyDcam exited')


    def ez_exposure_time(self, exposure_time):
        self.prop_setvalue(DCAM_IDPROP.EXPOSURETIME, exposure_time)


    def ez_triggersource_masterpluse(self, burst_times, interval):
        self.prop_setvalue(DCAM_IDPROP.TRIGGERSOURCE, DCAMPROP.TRIGGERSOURCE.MASTERPULSE)
        self.prop_setvalue(DCAM_IDPROP.MASTERPULSE_MODE, DCAMPROP.MASTERPULSE_MODE.BURST)
        self.prop_setvalue(DCAM_IDPROP.MASTERPULSE_TRIGGERSOURCE, DCAMPROP.MASTERPULSE_TRIGGERSOURCE.SOFTWARE)

        self.prop_setvalue(DCAM_IDPROP.MASTERPULSE_BURSTTIMES, burst_times)
        self.prop_setvalue(DCAM_IDPROP.MASTERPULSE_INTERVAL, interval)

    
    def ez_triggersource_external(self):
        self.prop_setvalue(DCAM_IDPROP.TRIGGERSOURCE, DCAMPROP.TRIGGERSOURCE.EXTERNAL)


    def ez_temperature(self):
        return self.prop_getvalue(DCAM_IDPROP.SENSORTEMPERATURE)
    

    def ez_roi(self, X0, Y0, W, H):
        self.prop_setvalue(DCAM_IDPROP.SUBARRAYHPOS, X0)
        self.prop_setvalue(DCAM_IDPROP.SUBARRAYVPOS, Y0)
        self.prop_setvalue(DCAM_IDPROP.SUBARRAYHSIZE, W)
        self.prop_setvalue(DCAM_IDPROP.SUBARRAYVSIZE, H)
        self.prop_setvalue(DCAM_IDPROP.SUBARRAYMODE,  2)


    def ez_wait_capture(self, timeout=int(2**31 - 1)):
        while True:
            if self.wait_event(DCAMWAIT_CAPEVENT.CYCLEEND, timeout) is not False:
                break
            dcamerr = self.lasterr()
            if dcamerr.is_timeout():
                raise TimeoutError('===: timeout')

    
    def ez_read_buf(self, iFrame, read_timestamp=True):
        frame = self.buf_getframe(iFrame)
        timestamp = self.ez_fmt_time(frame[0]) if read_timestamp else 0
        data = frame[1]
        return data, timestamp


    @classmethod
    def ez_fmt_time(self, buf_frame: DCAMBUF_FRAME):
        total_seconds = buf_frame.timestamp.sec + buf_frame.timestamp.microsec / 1_000_000
        china_time = datetime.fromtimestamp(total_seconds, tz=timezone.utc).astimezone(timezone(timedelta(hours=8)))
        return china_time.strftime('%Y-%m-%d %H:%M:%S.%f') 




class EasyALP4(ALP4):
    def __init__(self, version='4.3', libDir=os.path.join(f'{os.path.dirname(__file__)}', 'api/')):
        super().__init__(version, libDir)
        self.ez_isopen = False


    def __enter__(self):
        self.Initialize()
        self.ez_load_seq([self.ez_single_pixel(0)])
        self.Run(loop=False)
        self.Wait()
        self.ez_isopen = True
        return self


    def __exit__(self, *args):
        self.Halt()
        try:
            self.FreeSeq()
        except ValueError:
            pass
        self.Free()
        self.ez_isopen = False
        print('EasyALP4 exited')

    
    def ez_single_pixel(self, pixels):
        img = np.ones([self.nSizeY, self.nSizeX]) * (2**8 - 1)
        img[self.nSizeY//2 - pixels, self.nSizeX//2 + pixels] = 0
        return img.ravel()


    def ez_load_seq(self, Imgs, PictureTime=50):
        imgSeq = np.concatenate(Imgs)
        self.SeqAlloc(nbImg=len(Imgs), bitDepth=1)
        self.SeqPut(imgData=imgSeq)
        self.SeqControl(ALP_BIN_MODE, ALP_BIN_UNINTERRUPTED)
        self.SetTiming(pictureTime=PictureTime)



class RaspiLED:
    def __enter__(self):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(hostname='192.168.137.158', port=22, username='qlab', password='123456')
        print('SSH connected')
        self.turn_off()
        return self

    def turn_off(self):
        _ = self.ssh.exec_command('echo 2 > /sys/class/pwm/pwmchip2/unexport')

    def turn_on(self, duty):
        if duty > 100 or duty < 0:
            raise ValueError
        _ = self.ssh.exec_command(f'./pwm.sh 18 2000 {20 * duty}')

    def check(self, loops=1, speed=10):
        print("Trying to turn on LED, check camera's screen.")
        for _ in range(loops):
            for duty in range(1, 101):
                self.turn_on(duty)
                sleep(1/speed)
        self.turn_off()

    def __exit__(self, *args):
        self.turn_off()
        self.ssh.close()
        print('RaspiLED exited')