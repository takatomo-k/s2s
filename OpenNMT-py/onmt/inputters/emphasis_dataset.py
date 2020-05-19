import codecs
import numpy as np
import torch
import torch.nn as nn


class EmphasisTransform(object):
    def __init__(self, side, data, min_frequency):
        self.side=side
        cunt = {}
        #import pdb;pdb.set_trace()
        self._stoi = {'-2':0, '-1':1, '0':2, '1':3,'2':4}
        self._itos = {0:'-2', 1:'-1', 2:'0', 3:'1',4:'2'}
        self.lossfn=nn.NLLLoss(ignore_index=2)
    
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
        for i in text:
            if i in self._stoi:
                ret.append(self._stoi[i])
            else:
                ret.append(self._stoi['<unk>'])
        return torch.tensor(ret).unsqueeze(-1)
    
    def itos(self, idx):
        if isinstance(idx, list):
            return " ".join([self._itos[i.item()] for i in idx ])
        else: 
            return self._itos[idx.item()]

    def loss(self,hyp,ref):
        return self.lossfn(hyp,ref)