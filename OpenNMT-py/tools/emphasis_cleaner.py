import codecs
import argparse    # 1. argparseをインポート
parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')    # 2. パーサを作る
# 3. parser.add_argumentで受け取る引数を追加していく
parser.add_argument('--txt','-txt', help='この引数の説明（なくてもよい）')
args = parser.parse_args()    # 4. 引数を解析

for i in open(args.txt):
    txt=[]
    key,value=i.strip().split("@")
    flag=False
    for v in value.split():
        if v=="<":
            flag=True
        elif v==">":
            flag=False
        else:
            if v=="_":
                txt.append(str(2))
            elif "Normal" in key or not flag:
                txt.append(str(2))
            elif "Light" in key and flag:
                txt.append(str(0))
            elif "Medium" in key and flag:
                txt.append(str(1))
            elif "Strong" in key and flag:
                txt.append(str(3))
            elif "Very_strong" in key and flag:
                txt.append(str(4))
    print(key+"@"+" ".join(txt))
