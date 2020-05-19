from text import text_to_sequence, sequence_to_text, clean_text
import argparse
import codecs
import re
parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')    # 2. パーサを作る
parser.add_argument('-txt', help='この引数の説明（なくてもよい）')
parser.add_argument('-type',default="english_cleaners", help='この引数の説明（なくてもよい）')    # 必須の引数を追加
args = parser.parse_args()    # 4. 引数を解析

#import pdb; pdb.set_trace()
_abbreviations = [(re.compile('%s' % x[0], re.IGNORECASE), x[1]) for x in [
  ("'s ", " 's "),
  ("'s$", " 's"),
  ("'m ", " 'm "),
  ("'m$", " 'm"),
  ("'d ", " 'd "),
  ("'d$", " 'd"),
  ("'ll ", " 'll "),
  ("'ll$", " 'll"),
  ("'ve ", " 've "),
  ("'ve$", " 've"),
  ("'re ", " 're "),
  ("'re$", " 're"),
  ("n\'t ", " n't "),
  ("n\'t$", " n't"),

]]
numbers={1:"いち",2:"に",3:"さん",4:"よん",5:"ご",6:"ろく",7:"なな",8:"はち",9:"きゅう"}
    

def expand_abbreviations(text):
    #import pdb; pdb.set_trace()    
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text
check_dict={}
def check(text):
    for i in text.split():
        if "'" in i and i not in check_dict:
            check_dict.update({i:1})
        elif i in check_dict:
            check_dict[i]+=1

    return text


def number(num):
    #import pdb; pdb.set_trace()
    man=""
    if num=="０":
        return "ぜろ"
    if isinstance(num,str) and not num.isdecimal():
        return num
    else:
        if isinstance(num,str):
            i=int(num.replace(",",""))
        else:
            i=num
        if i > 10000:
            #import pdb; pdb.set_trace()
            man =number(int(i/10000))+" まん"
            #man=numbers[int((i%100000)/10000)]+" まん" if int((i%100000)/10000)>0 else ""
        sen=numbers[int((i%10000)/1000)].replace("いち","")+" せん" if int((i%10000)/1000) else ""
        hyk=numbers[int((i%1000)/100)].replace("いち","")+" ひゃく" if int((i%1000)/100) else ""
        juh=numbers[int((i%100)/10)].replace("いち","")+" じゅう" if int((i%100)/10) else ""
        ich=numbers[int(i%10)] if int(i%10) >0 else ""
        return (man+" "+sen+" "+hyk+" "+juh+" "+ich)

#ja_out=codecs.open("train.ja","w",encoding='utf8')
#hi_out=codecs.open("train.hi","w",encoding='utf8')

#import pdb; pdb.set_trace()
for i in codecs.open(args.txt,"r",encoding='utf8'):
    key, text = i.strip().split("@")
    #text=text.replace(" ","").replace("_"," ")
    text=clean_text(text,["english_cleaners"]).replace("-"," ").replace("?"," ?").replace("!"," !").replace(":"," ").replace(";"," ").replace("  "," ").replace(",","")#transliteration_cleaners,
    #print(key+"@"+" ".join(list(text.replace(" ","_"))))
    text=expand_abbreviations(text).replace(".","")
    text=text.replace(" ","_")
    print(key+"@"+" ".join(list(text)))
    #print(" ".join([number(i) for i in (text.split())]))

    #print(text)
#    text=expand_abbreviations(text)
#    text=check(text)
#    print(text)
    #print(key+"@"+" ".join(list(text.replace(" ","_"))))
    #print(" ".join(list(text.replace(" ","_"))))
    #han=[]
    #hira=[]
    #for j in i.strip().split():
    #    ja,hi=j.split("/")
    #    if hi !="UNK":
    #        han.append(ja)
    #        hira.append(hi)
    #ja_out.write(" ".join(han)+"\n")
    #hi_out.write(" ".join(hira)+"\n")
    
#print(check_dict)
#print(len(check_dict))
"""
train={}
for i in codecs.open("../../../en/text/word/train", "r", encoding='utf8'):
    key,en=i.strip().split("@")
    train[en]=key

valid={}
for i in codecs.open("../../../en/text/word/valid", "r", encoding='utf8'):
    key,en=i.strip().split("@")
    valid[en]=key
t=codecs.open("./train.hi", "w", encoding='utf8')
v=codecs.open("./valid.hi", "w", encoding='utf8')
g=[codecs.open("./test.en", "w", encoding='utf8'),codecs.open("./test.hi", "w", encoding='utf8')]
idx=1
for e,j in zip(codecs.open("./tmp.en", "r", encoding='utf8'),codecs.open("./tmp.hi", "r", encoding='utf8')):
    e=e.strip()
    j=j.strip()
    if len(e)<1 or len(j)<1:
        continue
    if e in train:
        t.write(train[e]+"@"+j+"\n")
    elif e in valid:
        v.write(valid[e]+"@"+j+"\n")
    else:
        g[0].write("ted_gtts_tst_ja_"+str(idx)+"@"+e+"\n")
        g[1].write("ted_gtts_tst_ja_"+str(idx)+"@"+j+"\n")
        idx+=1
#print(len(t),len(v),len(g))
"""