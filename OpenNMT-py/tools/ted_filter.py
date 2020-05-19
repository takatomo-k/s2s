import argparse
import codecs
import re
import random
parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')    # 2. パーサを作る
parser.add_argument('-src', help='この引数の説明（なくてもよい）')
parser.add_argument('-tgt',default="english_cleaners", help='この引数の説明（なくてもよい）')
args = parser.parse_args()

"""
src={}
pair={}
for i in open(args.src):
    key,value=i.strip().split("@")
    if value not in src or "BTEC" not in key:
        src[value] = key
    if key not in pair:
        pair[key]=value



for i in open(args.tgt):
    key,value=i.strip().split("@")
    if key in pair:
        if pair[key] in src:
            print(src[pair[key]]+"@"+value)
    print(key+"@"+value)
"""

src={}
for i in open(args.src):
    key,value,length=i.strip().split("@")
    src[key]=value+"@"+length

vs=open("valid_mfcc","w")
tt=open("test_mfcc","w")
ts=open("train_mfcc","w")

for i in open("train/char"):
    key,value = i.strip().split("@")
    if key in src:
        ts.write(key+"@"+src[key]+"\n")

for i in open("valid/char"):
    key,value = i.strip().split("@")
    if key in src:
        vs.write(key+"@"+src[key]+"\n")

for i in open("test/char"):
    key,value = i.strip().split("@")
    if key in src:
        tt.write(key+"@"+src[key]+"\n")


