# -*- coding: utf-8 -*-
import glob
import os
import codecs
import math
import torch.nn as nn
import tqdm
import random
from nnmnkwii import preprocessing as P
#from multiprocessing_generator import ParallelGenerator
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import Counter, defaultdict
from itertools import chain, cycle
import numpy as np
import torch.nn.functional as F
import librosa
from onmt.utils.logging import logger
from torch.utils.data import Dataset, Sampler, DataLoader
import gc
from multiprocessing import Manager
from multiprocessing import Pool
from onmt.utils.cleaners import str2cleaners

class TextTransform(object):
    def __init__(self, side, data, domain, lang):
        self.segment=domain
        self.min_frequency=3
        self.side=side
        self.cleaners=str2cleaners[lang]
        self._stoi = {'<pad>':0, '<s>':1, '</s>':2, '<unk>':3}
        self._itos = {0:'<pad>', 1:'<s>', 2:'</s>', 3:'<unk>'}    
        self.pad_idx = self._stoi['<pad>']
        self.bos_idx = self._stoi['<s>']
        self.eos_idx = self._stoi['</s>']
        self.unk_idx = self._stoi['<unk>']
        self.max_length= 200 
        self.dim=1
    
    def load(self,data, training):
        print('Loading Text')
        cunt={}
        examples={}
        lengths={}
        uniq={}
        #import pdb; pdb.set_trace()
        flag=True
        for sent in tqdm.tqdm(data,total=len(data)):
            key,txt,_=sent.strip().split('@')
            if txt not in uniq:
                uniq[txt]=self.cleaners(txt).split()
                
        if self.segment=='sp':
            pass
        elif self.segment=='chr':
            for key in uniq.keys():
                uniq[key]=list('_'.join(uniq[key]))
                if flag:
                    #print(uniq[key])
                    flag=False
        if training:
            for v in uniq.values():
                for word in v:
                    if word in cunt:
                        cunt[word] += 1
                    else:
                        cunt.update({word:1})
            
            keys = [i for i in cunt.keys()]
            keys.sort()
            for k in keys:
                #print(k)
                self._stoi[k]=len(self._stoi)
                self._itos[self._stoi[k]]=k
            print(len(self._stoi))
            print(self._stoi)
            self.dim=len(self._stoi)
        for sent in data:
            key,txt,_=sent.strip().split('@')
            if len(uniq[txt])< self.max_length:
                examples[key]=uniq[txt]
                lengths[key]=len(uniq[txt])
        return examples,lengths

    def __len__(self):
        return len(self._stoi)
    
    def __call__(self, inputs):
        if isinstance(inputs, str):
            return self.stoi(inputs.split())
        elif isinstance(inputs, list):
            try:
                if isinstance(inputs[0], str):
                    return self.stoi(inputs)
                elif isinstance(inputs[0], int):
                    return self.itos(torch.tensor(inputs))
                elif isinstance(inputs[0], tensor):
                    return self.itos(inputs)
            except:
                print(inputs)
        else:
            pass

    def stoi(self, text):
        if self.side!='src':
            text=['<s>']+text+['</s>']
        ret=[]
        last_txt=None
        for i in text:
            if i=='_' and last_txt=='_':
                continue
            if i in self._stoi:
                #print(i, self._stoi[i])
                ret.append(self._stoi[i])
            else:
                ret.append(self._stoi['<unk>'])
        
        return torch.tensor(ret).unsqueeze(-1)
    
    def itos(self, idx):
        try:   
            return [self._itos[i] for i in idx ]
        except:
            #import pdb;pdb.set_trace()
            return self._itos[idx.item()]
    def reverse(self,hyp):
        try:
        
            return self.itos(hyp[:,0])
        except:
            return self.itos(hyp)
