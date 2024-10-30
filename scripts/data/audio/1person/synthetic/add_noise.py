from __future__ import print_function, division, absolute_import

import os
import itertools

from tqdm import tqdm
from glob import glob
import numpy as np
import scipy.io.wavfile
import fnmatch
import warnings
from scipy.signal import lfilter

# Utility functions for reading and writing WAV files
def wavread(filename):
    fs, x = scipy.io.wavfile.read(filename)
    if np.issubdtype(x.dtype, np.integer):
        x = x / np.iinfo(x.dtype).max
    return x, fs

def wavwrite(filename, s, fs):
    if s.dtype != np.int16:
        s = np.array(s * 2**15, dtype=np.int16)
    if np.any(s > np.iinfo(np.int16).max) or np.any(s < np.iinfo(np.int16).min):
        warnings.warn('Warning: clipping detected when writing {}'.format(filename))
    scipy.io.wavfile.write(filename, fs, s)

def recursive_glob(treeroot, pattern):
    results = []
    for base, dirs, files in os.walk(treeroot):
        goodfiles = fnmatch.filter(files, pattern)
        results.extend(os.path.join(base, f) for f in goodfiles)
    return results

# Utility functions for adding noise and reverb
def rms_energy(x):
    return 10*np.log10((1e-12 + x.dot(x))/len(x))

def asl_meter(x, fs, nbits=16):
    '''Measure the Active Speech Level (ASR) of x following ITU-T P.56.
    If x is integer, it will be scaled to (-1, 1) according to nbits.
    '''

    if np.issubdtype(x.dtype, np.integer):
        x = x / 2**(nbits-1)

    # Constants
    MIN_LOG_OFFSET = 1e-20
    T = 0.03                # Time constant of smoothing in seconds
    g = np.exp(-1/(T*fs))
    H = 0.20                # Time of handover in seconds
    I = int(np.ceil(H*fs))
    M = 15.9                # Margin between threshold and ASL in dB

    a = np.zeros(nbits-1)                       # Activity count
    c = 0.5**np.arange(nbits-1, 0, step=-1)     # Threshold level
    h = np.ones(nbits)*I                        # Hangover count
    s = 0
    sq = 0
    p = 0
    q = 0
    asl = -100

    L = len(x)
    s = sum(abs(x))
    sq = sum(x**2)
    dclevel = s/np.arange(1, L+1)
    lond_term_level = 10*np.log10(sq/np.arange(1, L+1) + MIN_LOG_OFFSET)
    c_dB = 20*np.log10(c)

    for i in range(L):
        p = g * p + (1-g) * abs(x[i])
        q = g * q + (1-g) * p

        for j in range(nbits-1):
            if q >= c[j]:
                a[j] += 1
                h[j] = 0
            elif h[j] < I:
                a[j] += 1;
                h[j] += 1

    a_dB = -100 * np.ones(nbits-1)

    for i in range(nbits-1):
        if a[i] != 0:
            a_dB[i] = 10*np.log10(sq/a[i])

    delta = a_dB - c_dB
    idx = np.where(delta <= M)[0]

    if len(idx) != 0:
        idx = idx[0]
        if idx > 1:
            asl = bin_interp(a_dB[idx], a_dB[idx-1], c_dB[idx], c_dB[idx-1], M)
        else:
            asl = a_dB[idx]

    return asl

def bin_interp(upcount, lwcount, upthr, lwthr, margin, tol=0.1):
    n_iter = 1
    if abs(upcount - upthr - margin) < tol:
        midcount = upcount
    elif abs(lwcount - lwthr - margin) < tol:
        midcount = lwcount
    else:
        midcount = (upcount + lwcount)/2
        midthr = (upthr + lwthr)/2
        diff = midcount - midthr - margin
        while abs(diff) > tol:
            n_iter += 1
            if n_iter > 20:
                tol *= 1.1
            if diff > tol:
                midcount = (upcount + midcount)/2
                midthr = (upthr + midthr)/2
            elif diff < -tol:
                midcount = (lwcount + midcount)/2
                midthr = (lwthr + midthr)/2
            diff = midcount - midthr - margin
    return midcount

def add_noise(speech, noise, fs, snr, speech_energy='rms', asl_level=-26.0):
    '''Adds noise to a speech signal at a given SNR.
    The speech level is computed as the P.56 active speech level, and
    the noise level is computed as the RMS level. The speech level is considered
    as the reference.'''
    # Ensure masker is at least as long as signal
    if len(noise) < len(speech):
        noise = np.pad(noise, (0, len(speech) - len(noise)), 'constant')

    # Apply a fade-in effect to the noise
    fade_in_duration = int(0.1 * fs)  # 0.1 seconds fade-in
    fade_in = np.linspace(0, 1, fade_in_duration)
    noise[:fade_in_duration] = noise[:fade_in_duration] * fade_in

    # Generate random section of masker
    if len(noise) > len(speech):
        idx = np.random.randint(0, len(noise) - len(speech))
        noise = noise[idx:idx+len(speech)]

    # Ensure noise and speech are of the same length
    if len(noise) != len(speech):
        raise ValueError('len(noise) needs to be at least equal to len(speech)!')

    # Scale noise wrt speech at target SNR
    N_dB = rms_energy(noise)
    if speech_energy == 'rms':
        S_dB = rms_energy(speech)
    elif speech_energy == 'P.56':
        S_dB = asl_meter(speech, fs)
    else:
        raise ValueError('speech_energy has to be either "rms" or "P.56"')

    # Rescale N
    N_new = S_dB - snr
    noise_scaled = 10**(N_new/20) * noise / 10**(N_dB/20)

    y = speech + noise_scaled

    y = y/10**(asl_meter(y, fs)/20) * 10**(asl_level/20)

    # Chop off a few frames from the beginning of the mixed signal
    chop_duration = int(0.3 * fs)  # 0.3 seconds chop
    y = y[chop_duration:]

    return y.astype(np.int16), noise_scaled.astype(np.int16)

def add_reverb(speech, reverb, fs, speech_energy='rms', asl_level=-26.0):
    '''Adds reverb (convolutive noise) to a speech signal.
    The output speech level is normalized to asl_level.
    '''
    y = lfilter(reverb, 1, speech)
    y = y/10**(asl_meter(y, fs)/20) * 10**(asl_level/20)

    return y

class Dataset(object):
    '''Defines a corrupted speech dataset. Contains information about speech
    material, additive and convolutive noise sources, and how to store output.
    '''

    def __init__(self, speech_energy='P.56'):
        self.speech = list()
        self.noise = dict()
        self.reverb = dict()
        self.speech_energy = speech_energy

    def add_speech_files(self, path, recursive=False):
        '''Adds speech files to the dataset. If the path is for a file, adds a single
        file. Otherwise, adds WAV files in the specified folder. If recursive=True,
        adds all WAV files in the path recursively.
        '''
        if os.path.isfile(path):
            self.speech.append(path)
        elif os.path.isdir(path):
            if recursive:
                files = recursive_glob(path, '*.wav') + recursive_glob(path, '*.WAV')
            else:
                files = glob(os.path.join(path, '*.wav')) + glob(os.path.join(path, '*.WAV'))
            self.speech.extend(files)
        else:
            raise ValueError('Path needs to point to an existing file/folder')

    def _add_distortion_files(self, path, distortion_dict, name=None):
        '''Adds noise files to the dataset. path can be either for a single file or
        for a folder. name will replace the file name as a key in the noise file dict.
        '''
        if os.path.isfile(path):
            if name is None:
                name = os.path.splitext(os.path.basename(path))[0]
            distortion_dict[name] = path
        elif os.path.isdir(path):
            files = glob(os.path.join(path, '*.wav')) + glob(os.path.join(path, '*.WAV'))

            if name is not None:
                if type(name) != list or type(name) != tuple:
                    raise ValueError('When path is a folder, name has to be a list or tuple with the same length as the number of distortion files in the folder.')
                elif len(name) != len(files):
                    raise ValueError('len(name) needs to be equal to len(files)')
            else:
                name = [os.path.splitext(os.path.basename(f))[0] for f in files]

            for n, f in zip(name, files):
                distortion_dict[n] = f
        else:
            raise ValueError('Path needs to point to an existing file/folder')

    def add_noise_files(self, path, name=None):
        self._add_distortion_files(path, self.noise, name=name)

    def add_reverb_files(self, path, name=None):
        self._add_distortion_files(path, self.reverb, name=name)

    def generate_condition(self, snrs, noise, output_dir, reverb=None, files_per_condition=None):
        if noise not in self.noise.keys():
            raise ValueError('noise not in dataset')

        if type(snrs) is not list:
            snrs = [snrs]

        n, nfs = wavread(self.noise[noise])

        if reverb is not None:
            r, rfs = wavread(self.reverb[reverb])
            condition_name = '{}_{}'.format(reverb, noise)
        else:
            condition_name = noise

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        try:
            for snr in snrs:
                os.mkdir(os.path.join(output_dir, '{}_{}dB'.format(condition_name, snr)))
        except OSError:
            print('Condition folder already exists!')

        for snr in snrs:
            if files_per_condition is not None:
                speech_files = np.random.choice(self.speech, files_per_condition, replace=False).tolist()
            else:
                speech_files = self.speech

            for f in tqdm(speech_files, desc='{}dB'.format(snr)):
                x, fs = wavread(f)
                if fs != nfs:
                    raise ValueError('Speech file and noise file have different fs!')
                if reverb is not None:
                    if fs != rfs:
                        raise ValueError('Speech file and reverb file have different fs!')
                    x = add_reverb(x, r, fs, speech_energy=self.speech_energy)
                y, _ = add_noise(x, n, fs, snr, speech_energy=self.speech_energy)
                wavwrite(os.path.join(output_dir, '{}_{}dB'.format(condition_name, snr),
                    os.path.basename(f)), y, fs)

    def generate_dataset(self, snrs, output_dir, files_per_condition=None):
        if type(snrs) is not list:
            snrs = [snrs]

        if len(self.reverb) > 0:
            for reverb, noise in itertools.product(self.reverb.keys(), self.noise.keys()):
                self.generate_condition(snrs, noise, output_dir,
                        reverb=reverb,
                        files_per_condition=files_per_condition)
        else:
            for noise in self.noise.keys():
                self.generate_condition(snrs, noise, output_dir,
                    reverb=None,
                    files_per_condition=files_per_condition)

class SpeechNoiseMixer(Dataset):
    '''Class to mix clean speech files with noise files and generate mixed output files'''

    def __init__(self, speech_energy='P.56'):
        super(SpeechNoiseMixer, self).__init__(speech_energy=speech_energy)

    def mix_speech_and_noise(self, clean_folder, noise_folder, output_folder, snrs, recursive=False, files_per_condition=None):
        '''Mixes speech and noise files and saves the output in the specified folder.
        
        Args:
            clean_folder (str): Path to the folder with clean speech WAV files.
            noise_folder (str): Path to the folder with noise WAV files.
            output_folder (str): Path to the output folder to save mixed WAV files.
            snrs (list): List of SNRs (in dB) for mixing the noise with the speech.
            recursive (bool): If True, searches for WAV files recursively in the specified folder.
            files_per_condition (int): Number of files to use per condition. If None, use all files.
        '''
        # Add speech files
        self.add_speech_files(clean_folder, recursive=recursive)
        
        # Add noise files
        self.add_noise_files(noise_folder)
        
        # Generate dataset with mixed files
        self.generate_dataset(snrs, output_folder, files_per_condition=files_per_condition)

    def mix_single_file(self, clean_wav, noise_wav, output_wav, snr):
        '''Mixes a single speech file with a noise file at a specified SNR and writes the output.
        
        Args:
            clean_wav (str): Path to the clean speech WAV file.
            noise_wav (str): Path to the noise WAV file.
            output_wav (str): Path to save the noisy output WAV file.
            snr (float): SNR (in dB) for mixing the noise with the speech.
        '''
        x, fs = wavread(clean_wav)
        n, nfs = wavread(noise_wav)
        
        if fs != nfs:
            raise ValueError('Sampling rates of speech and noise files must be the same!')

        # Apply reverb if needed (modify if you want to add reverb handling)
        y, noise_scaled = add_noise(x, n, fs, snr, speech_energy=self.speech_energy)
        wavwrite(output_wav, y, fs)