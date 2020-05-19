# coding: utf-8

from itertools import chain, starmap
from collections import Counter

import torch
from torch.utils.data import Dataset, Sampler, DataLoader

class MySampler(Sampler):
    def __init__(self, source, batch_size, shuffle=False):
        self.source = np.reshape(source, (-1, batch_size))
    
    def __iter__(self):
        return iter(self.source)

    def __len__(self):
        return len(self.source)

class MyTransform(object):
    def __init__(self, examples):
        self.examples = examples
    
    def __call__(self, idx):
        #return dict(key,value.__getitem__(idx) for key, value in self.examples.items())
        pass
        
class MyDataset(Dataset):
    def __init__(self, data, fields, opt):
        #super(Dataset, self).__init__()
        examples = dict((key, DatasetBase(key, path, value,
                         max_length=opt.src_seq_length if key == 'src' else opt.tgt_seq_length) 
                         for path, (key, value) in enumerate(data, fields.items())))
        self.transform = MyTransform(examples)
    
    def __getitem__(self, idx):
        return self.transform(idx)


class DatasetBase(Dataset):
    def __init__(self, key, path, field, max_length):
        self.key = key
        self.examples = dict()
        for line in codecs.open(path, "r", encoding='utf8'):
            if len(line.strip().split("@"))==2:
                key, value = line.strip().split("@")
                value = value.split()
                length = len(value)
            else:
                key, value, length = line.strip().split("@")
            if length < max_lengths:
                self.examples.update({key: value})
        self.transform = field
    
    def __getitem__(self, idx):
        return self.transform(self.examples[idx])

class TextTransform(object):
    def __init__(self, path, opt, src=False):
        cunt = {}
        for sent in codecs.open(path, "r", encoding='utf8'):
            for word in sent.stirp().split("@")[1]:
                if word in vocab:
                    cunt.update({word:1})
                else:
                    cunt[word] += 1
        self._stoi = {'<pad>':0, '<s>':1, '</s>':2, '<unk>':3}
        self._itos = {0:'<pad>', 1:'<s>', 2:'</s>', 3:'<unk>'}
        for key,value in cunt.items():
            if value > min_frequency:
                self._stoi.update({key:len(self.vocab)})
                self._itos.update({len(self.vocab):key})
        self.pad_idx = self.vocab['<pad>']
        self.bos_idx = self.vocab['<s>']
        self.eos_idx = self.vocab['</s>']
        self.unk_idx = self.vocab['<unk>']

    def __len__(self):
        return len(self.vocab)
    
    def __call__(self, inputs):
        if isinstance(inputs, tensor):
            return self.itos(inputs)
        elif isinstance(inputs, str):
            return self.stoi(inputs.split())
        elif isinstance(inputs, list):
            if isinstance(inputs[0], str):
                return self.stoi(inputs)
            elif isinstance(inputs[0], int):
                return self.itos(torch.tensor(inputs))
            elif isinstance(inputs[0], tensor):
                return self.itos(inputs)
        else:
            pass

    def stoi(self, text):
        return [self._sto1(i) for i in text]

    def itos(self, idx):
        return [self._itos[i.item()] for i in idx ]

class AudioTransform(object):
    def __init__(self, path, opt):
        for line in codecs.open(path, "r", encoding='utf8'):
            audio_path = line.strip().split("@")[1]
            self.dim=np.load(audio_path).shape[-1]

    def __len__(self):
        return self.dim

    def __call__(self, inputs):
        feat = torch.from_numpy(np.load(audio_path))
        return feat

class MelTransform(object):
    def __init__(self, path, opt):
        for line in codecs.open(path, "r", encoding='utf8'):
            audio_path = line.strip().split("@")[1]
            self.dim=np.load(audio_path).shape[-1]*5

    def __len__(self):
        return self.dim

    def __call__(self, inputs):
        feat = torch.from_numpy(np.load(audio_path))
        return feat