import librosa
import librosa.filters
import math
import numpy as np
import scipy

from scipy.io import wavfile
from scipy import signal
import os, copy
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font_path = '/home/is/takatomo-k/Desktop/IPAexfont00401/ipaexg.ttf'
font_prop = FontProperties(fname=font_path,size=20)

plt.switch_backend('agg')
plt.rcParams["font.size"] = 14

# Audio:
num_mels=80
n_fft=2048
num_freq=1025
sample_rate=16000
frame_length_ms=50
frame_shift_ms=12.5
_preemphasis=0.97
min_level_db=-100
ref_level_db=20

frame_shift = 0.0125 # seconds
frame_length = 0.05 # seconds
hop_length = int(sample_rate*0.0125) # samples.
win_length = int(sample_rate*0.05) # samples.

# Eval:
max_iters=200
griffin_lim_iters=60
power=1.5              # Power to raise magnitudes to prior to Griffin-Lim

def get_spectrograms(fpath):
    '''Parse the wave file in `fpath` and
    Returns normalized melspectrogram and linear spectrogram.
    Args:
      fpath: A string. The full path of a sound file.
    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
      mag: A 2d array of shape (T, 1+n_fft/2) and dtype of float32.
    '''
    # Loading sound file
    y, sr = librosa.load(fpath, sr=sample_rate)
    #sr,y=wavfile.read(fpath)
    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - _preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(sample_rate, n_fft, num_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel)) - ref_level_db
    mag = 20 * np.log10(np.maximum(1e-5, mag)) - ref_level_db 
    #import pdb; pdb.set_trace()
    # normalize
    mel = np.clip((mel - min_level_db) / -min_level_db, 1e-8, 1)
    mag = np.clip((mag - min_level_db) / -min_level_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag

def plot_spectrograms(feat):
    fig = plt.figure()
    return plt.specgram(feat,Fs=sample_rate,noverlap=n_fft-hop_length,mode='magnitude')
    


def spectrogram2wav(mag):
    '''# Generate wave file from linear magnitude spectrogram
    Args:
      mag: A numpy array of (T, 1+n_fft//2)
    Returns:
      wav: A 1-D numpy array.
    '''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * -min_level_db) + min_level_db

    # to amplitude
    mag += ref_level_db
    mag = np.power(10.0, mag * 0.05) 

    # wav reconstruction
    wav = griffin_lim(mag**power)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -_preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)
    #import pdb; pdb.set_trace()
    return wav.astype(np.float32)

def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.'''
    X_best = copy.deepcopy(spectrogram)
    for i in range(griffin_lim_iters):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, n_fft, hop_length, win_length=win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y

def invert_spectrogram(spectrogram):
    '''Applies inverse fft.
    Args:
      spectrogram: [1+n_fft//2, t]
    '''
    return librosa.istft(spectrogram, hop_length, win_length=win_length, window="hann")


def load_wav(path):
  return librosa.core.load(path, sr=sample_rate)[0]


def save_wav(wav, path):
  #wav = wav / max(0.01, np.max(np.abs(wav)))*20000
  scipy.io.wavfile.write(path, sample_rate, wav.astype(np.int16))


def preemphasis(x):
  return scipy.signal.lfilter([1, -_preemphasis], [1], x)


def inv_preemphasis(x):
  return scipy.signal.lfilter([1], [1, -_preemphasis], x)


def spectrogram(y,fs=16000):
  D = _stft(preemphasis(y),fs)
  S = _amp_to_db(np.abs(D)) - ref_level_db
  return _normalize(S)


def inv_spectrogram(spectrogram):
  '''Converts spectrogram to waveform using librosa'''
  S = _db_to_amp(_denormalize(spectrogram) + ref_level_db)  # Convert back to linear
  return inv_preemphasis(_griffin_lim(S ** power))          # Reconstruct phase


def inv_spectrogram_tensorflow(spectrogram):
  '''Builds computational graph to convert spectrogram to waveform using TensorFlow.
  Unlike inv_spectrogram, this does NOT invert the preemphasis. The caller should call
  inv_preemphasis on the output after running the graph.
  '''
  S = _db_to_amp_tensorflow(_denormalize_tensorflow(spectrogram) + ref_level_db)
  return _griffin_lim_tensorflow(tf.pow(S, power))


def melspectrogram(fs,y):
  D = _stft(preemphasis(y),fs)
  S = _amp_to_db(_linear_to_mel(np.abs(D),fs)) - ref_level_db
  return _normalize(S)

def mel_liner(fs,y):
    D = _stft(preemphasis(y),fs)
    mel = _amp_to_db(_linear_to_mel(np.abs(D),fs)) - ref_level_db
    linear = _amp_to_db(np.abs(D)) - ref_level_db
    return _normalize(mel),_normalize(linear)
    
def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
  window_length = int(sample_rate * min_silence_sec)
  hop_length = int(window_length / 4)
  threshold = _db_to_amp(threshold_db)
  for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x:x+window_length]) < threshold:
      return x + hop_length
  return len(wav)


def _griffin_lim(S):
  '''librosa implementation of Griffin-Lim
  Based on https://github.com/librosa/librosa/issues/434
  '''
  angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
  S_complex = np.abs(S).astype(np.complex)
  y = _istft(S_complex * angles)
  for i in range(griffin_lim_iters):
    angles = np.exp(1j * np.angle(_stft(y)))
    y = _istft(S_complex * angles)
  return y


def _griffin_lim_tensorflow(S):
  '''TensorFlow implementation of Griffin-Lim
  Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb
  '''
  with tf.variable_scope('griffinlim'):
    # TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
    S = tf.expand_dims(S, 0)
    S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
    y = _istft_tensorflow(S_complex)
    for i in range(griffin_lim_iters):
      est = _stft_tensorflow(y)
      angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
      y = _istft_tensorflow(S_complex * angles)
    return tf.squeeze(y, 0)


def _stft(y,fs=16000):
  n_fft, hop_length, win_length = _stft_parameters(fs)
  return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y,fs=16000):
  _, hop_length, win_length = _stft_parameters(fs)
  return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _stft_tensorflow(signals,fs):
  n_fft, hop_length, win_length = _stft_parameters(fs)
  return tf.contrib.signal.stft(signals, win_length, hop_length, n_fft, pad_end=False)


def _istft_tensorflow(stfts,fs):
  n_fft, hop_length, win_length = _stft_parameters(fs)
  return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)


def _stft_parameters(fs=16000):
  n_fft = (num_freq - 1) * 2
  hop_length = int(frame_shift_ms / 1000 * fs)
  win_length = int(frame_length_ms / 1000 * fs)
  return n_fft, hop_length, win_length


# Conversions:

_mel_basis = None

def _linear_to_mel(spectrogram,fs=16000):
  global _mel_basis
  if _mel_basis is None:
    _mel_basis = _build_mel_basis(fs)
  return np.dot(_mel_basis, spectrogram)

def _build_mel_basis(fs=16000):
  n_fft = (num_freq - 1) * 2
  return librosa.filters.mel(fs, n_fft, n_mels=num_mels)

def _amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
  return np.power(10.0, x * 0.05)

def _db_to_amp_tensorflow(x):
  return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)

def _normalize(S):
  return np.clip((S - min_level_db) / -min_level_db, 0, 1)

def _denormalize(S):
  return (np.clip(S, 0, 1) * -min_level_db) + min_level_db

def _denormalize_tensorflow(S):
  return (tf.clip_by_value(S, 0, 1) * -min_level_db) + min_level_db


if __name__ == 'main':
  import glob
  data= glob.glob('/project/nakamura-lab08/Work/takatomo-k/data/raw/ja/btec/generated/mag/*')
  