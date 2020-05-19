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
from onmt.inputters.audio_dataset import AudioTransform
from onmt.inputters.text_dataset import TextTransform
def collate_fn(data):
    keys=[i for i in data[0].keys()]
    batch=Batch()
    for key in keys:
        feat = [d[key] for d in data]
        if key =="key":
            setattr(batch, key, feat)
        else:
            length = torch.tensor([d.shape[0] for d in feat])
            setattr(batch, key, (pad_sequence(feat),length))
    setattr(batch, 'batch_size', length.shape[0])
    setattr(batch, 'indices', torch.tensor([i for i in range(length.shape[0])]))
    return batch

def pairwise(examples):
    org_len=None
    while True:
            flag=True
            for src_key in examples.keys():
                if org_len==None:
                    org_len=len(examples[src_key])
                for tgt_key in examples.keys():
                    if tgt_key!=src_key:
                        diff = examples[src_key].keys()^examples[tgt_key].keys()
                        if len(diff)!=0:
                            examples[src_key].filter(diff)
                            examples[tgt_key].filter(diff)
                            flag = False
            if flag:
                print("Pairwised examples:", org_len," -> ", len(examples[src_key]))
                return examples

def get_max_length(path):
    if "char" in path:
        return 200
    elif "bpe" in path:
        return 100
    elif "word" in path:
        return 50
    else:
        return 1000

class Batch(object):
    def __init__(self):
        pass
    
    def to(self,device):
        for key, value in self.__dict__.items():
            if isinstance(value, tuple):
                setattr(self, key, (value[0].to(device),value[1].to(device)))
            elif key in {"batch_size","key"}:
                pass
            else:
                setattr(self, key, value.to(device))

class MyDataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=collate_fn,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None,device=0):
        self.dataloader = DataLoader(dataset, batch_size, shuffle, sampler,
                 batch_sampler, num_workers, collate_fn,
                 pin_memory, drop_last, timeout,
                 worker_init_fn)
        self.device = device
    
    def __iter__(self):
        return self
    
    def __next__(self):
        batch = next(iter(self.dataloader))
        batch.to(self.device)
        return batch

class MySampler(Sampler):
    def __init__(self, source, batch_size,is_train=False,step=0):
        self.source = []
        batch=[]
        data=[i for i in source.keys()]
        _key=[i for i in source.examples.keys()]
        src_key,tgt_key=_key[0],_key[-1]
        for key in  sorted(data, key=lambda x: (source.examples[src_key].lengths[x], source.examples[tgt_key].lengths[x]) ,reverse=True):
                batch.append(key)
                if len(batch)==batch_size:
                    self.source.append(batch)
                    batch=[]
        if is_train and step>0:
            random.shuffle(self.source)
        self.source=[flatten for inner in self.source for flatten in inner]

    def __iter__(self):
        return iter(self.source)

    def __len__(self):
        return len(self.source)

class MyTransform(object):
    def __init__(self, ):
        pass
    def __call__(self, item):
        return item

class MyDataset(Dataset):
    def __init__(self, data, fields, opt, training=True):
        examples = dict((key, DatasetBase(data[key], value, training)) for  key, value in fields.items())
        self.examples = pairwise(examples)
        self.transform = MyTransform()
        self.key=[i for i in examples.keys()][0]
        self.length = len(examples[self.key])
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item=dict((key,value.__getitem__(idx)) for key, value in self.examples.items())
        item['key']=idx
        return self.transform(item)

    def keys(self):
        return self.examples[self.key].keys()
    
    def update(self,stats):
        import pdb; pdb.set_trace()
        #print("a")
    
    def remove(self,data):
        for key in self.examples.keys():
            self.examples[key].filter(data)
        self.length = len(self.examples[self.key])

class DatasetBase(Dataset):
    def __init__(self, data, field, training):
        self.examples,self.lengths = field.load(data,training)
        #print("MaxLen:",_max_length,"Filter:",field.max_length)
        print("Total examples:",len(data),"->",len(self.examples))
        self.transform = field
    
    def keys(self):
        return self.examples.keys()
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.transform(self.examples[idx])
    
    def filter(self,keys):
        for key in keys:
            self.examples.pop(key,"")
            self.lengths.pop(key,"")
    def update(self,key,value):
        import pdb; pdb.set_trace()







class AdversarialTransform(object):
    def __init__(self, side):
        self.side=side
    def __len__(self):
        return 1
    def __call__(self, inputs):
            if inputs [0]=='tts':
                return torch.zeros(1)
            else:
                return torch.ones(1)
#        return torch.ones((1)).type(torch.LongTensor)*self.spkr[inputs[0]]

