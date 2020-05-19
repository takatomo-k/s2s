import argparse
parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')    # 2. パーサを作る
parser.add_argument('-src',default="en", help='この引数の説明（なくてもよい）')
args = parser.parse_args()

for i in open(args.src):
    i=i.strip().split("@")[0]
    if "BTEC" in i or "NMT" in i:
        print(i+"@"+"TTS")
    else:
        spkr = i.split(".")[0]
        spkr = spkr.split("-")[0]
        print(i+"@"+spkr)