import codecs
import argparse
parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')    # 2. パーサを作る

# 3. parser.add_argumentで受け取る引数を追加していく
parser.add_argument('--src','-src', help='この引数の説明（なくてもよい）')
args = parser.parse_args()    # 4. 引数を解析
txt=[]
vocab={}
char ={}
for i in open(args.src):
    if "@" in i:
        i= i.strip().split("@")[1]
    
    for j in i.split():
        if j not in vocab:
            vocab[j]=0
    
    for j in list(i):
        if j not in char:
            char[j]=0
print(len(vocab))
print(len(char))