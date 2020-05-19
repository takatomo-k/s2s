import os
import tqdm
import sys
import numpy as np
import glob
import random
import argparse

parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')    # 2. パーサを作る
parser.add_argument('-src',default="en", help='この引数の説明（なくてもよい）')
parser.add_argument('-tgt',default="fr", help='この引数の説明（なくてもよい）')
args = parser.parse_args()

src={}
pair={}
for i in open("btec/train/word"):
    key,value=i.strip().split("@")
    src[key]=value
#import pdb;pdb.set_trace()
for i in open(args.src):
    key,value=i.strip().split("@")
    if key in src:
        pair[src[key]]=value

for i in open("natural_btec/train/word"):
    key,value=i.strip().split("@")
    if value in pair:
        print(key+"@"+pair[value])
