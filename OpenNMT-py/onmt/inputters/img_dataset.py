import sys,os
from torch.utils.data import Dataset
import librosa
import random
import numpy as np

class ImgDataset(Dataset):
    def __init__(self,train_src,train_tgt,valid_src,valid_tgt,vocab_src,vocab_tgt):
        super(ImgDataset, self).__init__()
        pass