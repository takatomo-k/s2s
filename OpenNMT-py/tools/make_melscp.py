import tqdm
import argparse
import glob
import os
import numpy as np
parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')    # 2. パーサを作る
parser.add_argument('-src',default="en", help='この引数の説明（なくてもよい）')
args = parser.parse_args()

#import pdb;pdb.set_trace()
out=open("mel","w")
for i in glob.glob(args.src+"*"):
    i=i.strip()
    try:
        tmp=np.load(i)
        out.write(os.path.basename(i).replace(".npy","")+"@"+i+"@"+str(tmp.shape[0])+"\n")
    except:
        print(i)