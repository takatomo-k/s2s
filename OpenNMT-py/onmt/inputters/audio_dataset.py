from __future__ import division, print_function, absolute_import
import torch.nn.functional as F
from docopt import docopt
import numpy as np
import torch
from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from nnmnkwii import preprocessing as P
from nnmnkwii.io import hts
from nnmnkwii.frontend import merlin as fe
import librosa
import librosa.filters
import math
import numpy as np
import scipy
from scipy.io import wavfile
from scipy import signal
import os, copy
import numpy as np
from multiprocessing import Pool
import pysptk
import pyworld
import pysptk
import pyworld
from scipy.io import wavfile
from tqdm import tqdm
from os.path import basename, splitext, exists, expanduser, join
import os
import sys
from glob import glob
import pyworld
from scipy.io import wavfile
from scipy import stats,mean
from os.path import basename, splitext, exists, expanduser, join
import os
import subprocess
import sys
from glob import glob
from gtts import gTTS
import codecs
from multiprocessing import Manager
import shutil
import tqdm
from nnmnkwii import preprocessing as P
from nnmnkwii import paramgen
from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from nnmnkwii.io import hts
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.postfilters import merlin_post_filter

# Acoustic features
order=29
frame_period=5
f0_floor=71.0
f0_ceil=700
use_harvest=True,  # If False, use dio and stonemask
windows=[
    (0, 0, np.array([1.0])),
    (1, 1, np.array([-0.5, 0.0, 0.5])),
    (1, 1, np.array([1.0, -2.0, 1.0])),
]
f0_interpolation_kind="quadratic"
mod_spec_smoothing=True
mod_spec_smoothing_cutoff=50  # Hz

recompute_delta_features=False

# Stream info
# (mgc, lf0, vuv, bap)
stream_sizes=[(order+1)*3, 3, 1, 3]
has_dynamic_features=[True, True, False, True]

# Streams used for computing adversarial loss
# NOTE: you should probably change discriminator's `in_dim`
# if you change the adv_streams
adversarial_streams=[True, False, False, False]
# Don't set the value > 0 unless you are sure what you are doing
# mask 0 to n-th mgc for adversarial loss
# e.g, for n=2, 0-th and 1-th mgc coefficients will be masked
mask_nth_mgc_for_adv_loss=2

# num_freq = 1024
n_fft = 2048
sr = 16000
# frame_length_ms = 50.
# frame_shift_ms = 12.5
preemphasis = 0.97
frame_shift = 0.0125 # seconds
frame_length = 0.05 # seconds
hop_length = int(sr*frame_shift) # samples.
win_length = int(sr*frame_length) # samples.
n_mels = 80 # Number of Mel banks to generate
power = 1.2 # Exponent for amplifying the predicted magnitude
min_level_db = -100
ref_level_db = 20
max_db = 100
ref_db = 20
n_iter = 60

class AudioTransform(object):
    def __init__(self,side, data, domain):
        self.side=side
        self.domain=domain
        self.max_length=1000
        self.dim=1

    def load(self, data,training=True):
        print('Loading Audio')
        self.training=training
        m=Manager()
        audio=set()
        for i in data:
            _,_,path =i.strip().split('@')
            if path not in audio:
                audio.add(path+self.domain+'.npy')
        self.feats=m.dict() 
        if training:
            self.mean,self.std=m.list(),m.list()
        with Pool(processes=100) as p:
            with tqdm.tqdm(total=len(audio)) as pbar:
                for i, _ in tqdm.tqdm(enumerate(p.imap_unordered(self._load, audio))):
                    pbar.update()
        examples,lengths={},{}
        for i in data:
            key,_,path =i.strip().split('@')
            path+=self.domain+'.npy'
            if path in self.feats:
                examples[key]=path
                lengths[key]=self.feats[path]
        #$import pdb; pdb.set_trace()
        
        if self.training:
            self.mean=(sum(self.mean)/(len(self.mean)+1))
            self.std =sum(self.std)/(len(self.std)+1)
            self.dim =self.mean.shape[0]
        
        del self.feats

        return examples,lengths

    def _load(self, path):
        try:
            feat=np.load(path).astype(np.float32)
            #np.save(path,feat)
            
            if feat.shape[0]< self.max_length:
                self.feats[path]=feat.shape[0]
                if self.training:
                    self.mean.append(feat.mean(axis=0))
                    self.std.append(feat.mean(axis=0))                
        except:
            #print('Error',path)
            pass

    def __len__(self):
        #import pdb; pdb.set_trace()
        #np.save("/home/is/takatomo-k/mean.npy",self.mean)
        #np.save("/home/is/takatomo-k/std.npy",self.std)
        if self.domain=='mel':
            return self.dim *3
        else:
            return self.dim
            
    def __call__(self, path):
        feat = np.load(path).astype(np.float32)
        if self.domain=='wld':
            #feat= P.scale(feat, self.mean, self.std)
            mean=feat.mean(axis=0)
            std=feat.std(axis=0)
            feat= P.scale(feat, mean, std)
            mean= np.vstack((mean, np.vstack((mean,mean))))
            std=  np.vstack((std, np.vstack((std,std))))
            feat = np.vstack((np.vstack((feat,mean)),std))
        elif self.domain=='mel':
            feat= (feat-self.mean)/(self.std*self.std)
            feat= torch.from_numpy(feat)
            
            if feat.shape[0]%3 !=0:
            #print(feat.shape,5-feat.shape[0]%5)
                feat = F.pad(feat,(0,0,0,3-feat.shape[0]%3))
            feat = feat.contiguous().view(-1,self.dim*3)
            feat = F.pad(feat,(0,0,1,0),value=1) #pad sos
            feat = F.pad(feat,(0,0,0,1)) #pad eos
            return feat.type(torch.FloatTensor)
        else:
            return torch.from_numpy(feat)
        
        
    def reverse(self,hyp):
        if self.domain=="mag":
            return self.spectrogram2wav(hyp)
        hyp=hyp.reshape(-1,self.mean.shape[0])
        self.std=hyp[-2]
        self.mean=hyp[-5]
        wav =self.gen_waveform(hyp[6:])
        return wav
    
    def gen_parameters(self, y_predicted, mge_training=False):
        mgc_dim, lf0_dim, vuv_dim, bap_dim = stream_sizes
        mgc_start_idx = 0
        lf0_start_idx = mgc_dim
        vuv_start_idx = lf0_start_idx + lf0_dim
        bap_start_idx = vuv_start_idx + vuv_dim
        # MGE training
        if mge_training:
            # Split acoustic features
            mgc = y_predicted[:, :lf0_start_idx]
            lf0 = y_predicted[:, lf0_start_idx:vuv_start_idx]
            vuv = y_predicted[:, vuv_start_idx]
            bap = y_predicted[:, bap_start_idx:]

            # Perform MLPG on normalized features
            mgc = paramgen.mlpg(mgc, np.ones(mgc.shape[-1]), windows)
            lf0 = paramgen.mlpg(lf0, np.ones(lf0.shape[-1]), windows)
            bap = paramgen.mlpg(bap, np.ones(bap.shape[-1]), windows)
            #import pdb; pdb.set_trace()
            # When we use MGE training, denormalization should be done after MLPG.
            mgc = P.inv_scale(mgc, self.mean[:mgc_dim // len(windows)],
                            self.std[:mgc_dim // len(windows)])
            lf0 = P.inv_scale(lf0, self.mean[lf0_start_idx:lf0_start_idx + lf0_dim // len(windows)],
                            self.std[lf0_start_idx:lf0_start_idx + lf0_dim // len(windows)])
            bap = P.inv_scale(bap, self.mean[bap_start_idx:bap_start_idx + bap_dim // len(windows)],self.std[bap_start_idx:bap_start_idx + bap_dim // len(windows)])
            vuv = P.inv_scale(vuv, self.mean[vuv_start_idx], self.std[vuv_start_idx])
        else:
            # Denormalization first
            y_predicted = P.inv_scale(y_predicted, self.mean, self.std)

            # Split acoustic features
            mgc = y_predicted[:, :lf0_start_idx]
            lf0 = y_predicted[:, lf0_start_idx:vuv_start_idx]
            vuv = y_predicted[:, vuv_start_idx]
            bap = y_predicted[:, bap_start_idx:]

            # Perform MLPG
            Y_var = self.std * self.std
            mgc = paramgen.mlpg(mgc, Y_var[:lf0_start_idx], windows)
            lf0 = paramgen.mlpg(lf0, Y_var[lf0_start_idx:vuv_start_idx], windows)
            bap = paramgen.mlpg(bap, Y_var[bap_start_idx:], windows)

        return mgc, lf0, vuv, bap

    def spectrogram2wav(self, mag):
        '''# Generate wave file from linear magnitude spectrogram
        Args:
        mag: A numpy array of (T, 1+n_fft//2)
        Returns:
        wav: A 1-D numpy array.
        '''
        def griffin_lim(spectrogram):
            '''Applies Griffin-Lim's raw.'''
            X_best = copy.deepcopy(spectrogram)
            for i in range(n_iter):
                X_t = librosa.istft(X_best, hop_length, win_length=win_length, window="hann")
                est = librosa.stft(X_t, n_fft, hop_length, win_length=win_length)
                phase = est / np.maximum(1e-8, np.abs(est))
                X_best = spectrogram * phase
            X_t = librosa.istft(X_best, hop_length, win_length=win_length, window="hann")
            y = np.real(X_t)

            return y
        # transpose
        mag = mag.T

        # de-noramlize
        mag = (np.clip(mag, 0, 1) * max_db) - max_db + ref_db

        # to amplitude
        mag = np.power(10.0, mag * 0.05)

        # wav reconstruction
        wav = griffin_lim(mag**power)

        # de-preemphasis
        wav = signal.lfilter([1], [1, -preemphasis], wav)

        # trim
        wav, _ = librosa.effects.trim(wav)

        return wav.astype(np.float32)

    def gen_waveform(self, y_predicted, post_filter=False, coef=1.4,
                    fs=16000, mge_training=False):
        alpha = pysptk.util.mcepalpha(fs)
        fftlen = fftlen = pyworld.get_cheaptrick_fft_size(fs)

        # Generate parameters and split streams
        mgc, lf0, vuv, bap = self.gen_parameters(y_predicted, mge_training)

        if post_filter:
            mgc = merlin_post_filter(mgc, alpha, coef=coef)

        spectrogram = pysptk.mc2sp(mgc, fftlen=fftlen, alpha=alpha)
        aperiodicity = pyworld.decode_aperiodicity(bap.astype(np.float64), fs, fftlen)
        f0 = lf0.copy()
        f0[vuv < 0.5] = 0
        f0[np.nonzero(f0)] = np.exp(f0[np.nonzero(f0)])

        generated_waveform = pyworld.synthesize(f0.flatten().astype(np.float64),
                                                spectrogram.astype(np.float64),
                                                aperiodicity.astype(np.float64),
                                                fs, frame_period)
        # Convert range to int16
        generated_waveform = generated_waveform / \
            np.max(np.abs(generated_waveform)) #* 32767

        # return features as well to compare natural/genearted later
        return generated_waveform #, mgc, lf0, vuv, bap

    def recompute_delta_features(self, Y, has_dynamic_features=[True, True, False, True]):
        
        start_indices = np.hstack(([0], np.cumsum(stream_sizes)[:-1]))
        end_indices = np.cumsum(stream_sizes)
        static_stream_sizes = self.get_static_stream_sizes(
            stream_sizes, has_dynamic_features, len(windows))

        for start_idx, end_idx, static_size, has_dynamic in zip(
                start_indices, end_indices, static_stream_sizes, has_dynamic_features):
            if has_dynamic:
                
                y_static = Y[:, start_idx:start_idx + static_size]
                Y[:, start_idx:end_idx] = P.delta_features(y_static, windows)

        return Y
    
    def get_static_stream_sizes(self,stream_sizes, has_dynamic_features, num_windows):
        """Get static dimention for each feature stream.
        """
        static_stream_sizes = np.array(stream_sizes)
        static_stream_sizes[has_dynamic_features] = \
            static_stream_sizes[has_dynamic_features] / num_windows

        return static_stream_sizes