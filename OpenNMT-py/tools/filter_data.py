import tqdm
from multiprocessing import Pool
import argparse
import glob
import codecs
import numpy as np
import os
from multiprocessing import Process, Manager

valid=set([i.split("@")[0] for i in open("./valid/char")])
w =open("./valid_mfcc","w")
for i in open("./valid/mfcc"):
    if i.split("@")[0] in valid:
        w.write(i)

test=set([i.split("@")[0] for i in open("./test/char")])
j =open("./test_mfcc","w")
for i in open("./test/mfcc"):
    if i.split("@")[0] in test:
        j.write(i)

train_key={}
t=open("./train_char","w")
h=open("./train_mfcc","w")
for c,m in zip(open("./train/char"),open("./train/mfcc")):
    key,_ = c.strip().split("@")
    if key not in test and key not in valid:
        t.write(c)
        h.write(m)


