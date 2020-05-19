from googletrans import Translator

translator = Translator()
ref=open("ref.txt","w")
hyp=open("hyp.txt","w")

import argparse,os
import codecs
import re
parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')    # 2. パーサを作る
parser.add_argument('-src', help='この引数の説明（なくてもよい）')
parser.add_argument('-tgt',default="english_cleaners", help='この引数の説明（なくてもよい）')    # 必須の引数を追加
args = parser.parse_args()    # 4. 引数を解析

src={}
for i in open(args.src,"r"):
    key,value=i.strip().split("@")
    src[key]=value.replace(" ","").replace("_"," ")

tgt={}
for i in open(args.tgt,"r"):
    key,value=i.strip().split("@")
    tgt[key]=value.replace(" ","").replace("_"," ")

import tqdm
import pdb;pdb.set_trace()
for i in tqdm.tqdm(src.keys()):
    if i in tgt:
        try:
            result=translator.translate(src[i],dest='de')
            hyp.write(result.text.lower()+"\n")
            ref.write(tgt[i]+"\n")
        except:
            print(i)

os.system("perl tools/multi-bleu.perl "+"ref.txt" +" < "+"hyp.txt")