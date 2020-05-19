import glob
import subprocess
import tqdm
import Levenshtein
import os,sys
"""
for i in open("/home/is/takatomo-k/Desktop/Data/btec/hiragana/tmp"):
    i=i.strip().split()
    tmp=[]
    for j in i:
        j=j.split("/")[1]
        tmp.append(j)

    print(" ".join(tmp))
"""
import argparse    # 1. argparseをインポート

parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')    # 2. パーサを作る

# 3. parser.add_argumentで受け取る引数を追加していく
parser.add_argument('--dir','-dir', help='この引数の説明（なくてもよい）')
parser.add_argument('--seg','-seg', default="char", help='この引数の説明（なくてもよい）')
    # 必須の引数を追加
args = parser.parse_args()    # 4. 引数を解析

for _dir in glob.glob(args.dir+"/*"):
    if "events" in _dir:
        continue
    result_dir=_dir.replace("exp/log","exp/result")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    _ref=result_dir+"/ref.txt"
    _hyp=result_dir+"/hyp.txt"
    ref=open(_ref,"w")
    hyp=open(_hyp,"w")
    cer=0
    wer=0
    num=1

    with open(_dir+"/result.txt")as f:
        for line in f.readlines():
            if "ref:" in line:
                r=line.replace("ref:","")#.strip().split()
                
            if "hyp:" in line:
                h=line.replace("hyp:","")#.strip().split()

                if len(h) >1:
                    if "2char" in args.dir:
                        #import pdb; pdb.set_trace()
                        cer+=Levenshtein.distance(h,r)
                        r=r.replace(" ","").replace("_"," ")
                        h=h.replace(" ",'').replace("_"," ")
                        
                    elif "2sub" in args.dir:
                        r=r.replace("_ ","")
                        r=h.replace("_ ","")
                    
                    ref.write(r+"\n")
                    hyp.write(h+"\n")
                    wer+=Levenshtein.distance(h,r)
                    num+=1
    ref.close()
    hyp.close()
    print("########################")
    print(os.path.basename(result_dir))
    print("WER",wer/num,"CER",cer/num)
    os.system("perl tools/multi-bleu.perl "+_ref +" < "+_hyp)
    #os.system("perl tools/multi-bleu.perl "+_ref+"c" +" < "+_hyp+"c")
    print("########################")
    