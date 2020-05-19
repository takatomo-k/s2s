import argparse
import glob
parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')    # 2. パーサを作る
parser.add_argument('-src',default="en", help='この引数の説明（なくてもよい）')
parser.add_argument('-tgt',default="en", help='この引数の説明（なくてもよい）')
args = parser.parse_args()

char=set([i.split("@")[0] for i in open("char")])
for i in open("mfcc"):
    if i.split("@")[0] in char:
        print(i.strip())