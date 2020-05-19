import numpy as np
import argparse    # 1. argparseをインポート
import tqdm
import os
import glob
import codecs
import librosa
parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')    # 2. パーサを作る
"""
# 3. parser.add_argumentで受け取る引数を追加していく
parser.add_argument('--dir','-dir', help='この引数の説明（なくてもよい）')    # 必須の引数を追加
args = parser.parse_args()    # 4. 引数を解析
wav_list=[]
data_list=["all"]
for d in data_list:
    _list=open(args.dir+"/"+d).readlines()
    for i in tqdm.tqdm(_list,total=len(_list)):
        i=i.strip()
        tmp=np.load(i.split("@")[1])
        wav_list.append(i+"@"+str(tmp.shape[0]))
    with open(args.dir+"/"+d,"w") as f:
        for i in wav_list:
            f.write(i+"\n")
"""
parser.add_argument('--src','-src', help='この引数の説明（なくてもよい）')    # 必須の引数を追加
parser.add_argument('--tgt','-tgt', help='この引数の説明（なくてもよい）')    # 必須の引数を追加

args = parser.parse_args()    # 4. 引数を解析

wavs={}
f=codecs.open("./scp/mel","w")
#import pdb;pdb.set_trace()
for i in open(glob.glob(args.src+"/*")):
    try:
        feat=np.load(i)
        f.write(os.path.basename(i).replace(".npy","")+"@"+i+"@"+str(feat.shape[0])+"\n")
        #print(os.path.basename(i).replace(".npy","")+"@"+i+"@"+str(feat.shape[0]))
    except:
        pass