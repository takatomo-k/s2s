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
from scipy.io import wavfile
from scipy import stats,mean
from os.path import basename, splitext, exists, expanduser, join
import os
import subprocess
import sys
from glob import glob
from gtts import gTTS
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
text_data={}
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

def tts(key):
    res=gTTS(data[key],lang='fr')
    res.save('audio/mp3/'+key+".mp3")


def feat(key):
    try:
        mel,mag=get_spectrograms(key)
    except:
        print("feat error:",key)
    try:
        np.save('audio/mel/'+os.path.basename(key).replace(".wav",".npy"), mel)
        np.save('audio/mag/'+os.path.basename(key).replace(".wav",".npy"), mag)
        
    except:
        print("save error",'mel/'+os.path.basename(key).replace(".wav",".npy"))
    #np.save('audio/mag/'+key,mag)

def sox(path):
    try:
        cmd="sox " +path+ " -r 16000 "+path.replace("mp3","wav") #+" tempo 1.25 "
        subprocess.call(cmd .strip().split(" ") )
    except:
        print("sox error:",key)

lang="en"

if __name__ == "__main__":
    import tqdm
    import argparse
    import glob
    import codecs

    parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')    # 2. パーサを作る
    parser.add_argument('-src',default="fr", help='この引数の説明（なくてもよい）')
    parser.add_argument('-lang',default="fr", help='この引数の説明（なくてもよい）')
    args = parser.parse_args()
    #lang=args.lang
    #wavs=set(glob.glob("audio/wav/*"))
    
    files=codecs.open(args.src,'r',encoding='utf8').readlines()
    #mels=set(glob.glob("audio/mel/*"))
    
    data={}
    for i in tqdm.tqdm(files,total=len(files)):
        l,w=i.strip().split("@")
        if "TED-NMT" in l and "audio/wav/"+l+".wav" not in wavs:
            data[l]=w.replace(" ' ","'").replace(" ?","?")
    
    with Pool(processes=30) as p:
        with tqdm.tqdm(total=len(data)) as pbar:
            for i, _ in tqdm.tqdm(enumerate(p.imap_unordered(tts, [i for i in data.keys()]))):
                pbar.update()
    """
    data=[]
    mp3s=glob.glob("audio/mp3/*")
    
    for i in mp3s:
        if i.replace(".mp3",".npy").replace("mp3","mel") not in mels:
            data.append(i)

    with Pool(processes=30) as p:
        with tqdm.tqdm(total=len(data)) as pbar:
            for i, _ in tqdm.tqdm(enumerate(p.imap_unordered(sox, data))):
                pbar.update()
    
    
    data={}
    
    for i in glob.glob("/project/nakamura-lab05/Work/novitasari-s/speechchain/dataset/tedlium_release2/cutwav/*"):
        if "audio/mel/"+os.path.basename(i).replace(".wav",".npy").replace("wav","mel") not in mels:
            if i not in data:
                data[i]=0
            else:
                import pdb;pdb.set_trace()

    """                    
    #import pdb;pdb.set_trace()
    data=glob.glob(args.src+"*")
    with Pool(processes=30) as p:
        with tqdm.tqdm(total=len(data)) as pbar:
            for i, _ in tqdm.tqdm(enumerate(p.imap_unordered(feat, data))):
                pbar.update()
    

    """  
    data={}
    for i in tqdm.tqdm(files,total=len(files)):
        l,w=i.strip().split("@")
        data[l]=w
    
    for i in mels:
        i=os.path.basename(i).replace(".npy","")
        if i not in data:
            print(i)
    """