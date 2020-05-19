from text import text_to_sequence, sequence_to_text, clean_text
import codecs
import glob
import argparse
import re
import os
parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')    # 2. パーサを作る
parser.add_argument('-src',default="en", help='この引数の説明（なくてもよい）')
parser.add_argument('-tgt',default="fr", help='この引数の説明（なくてもよい）')
args = parser.parse_args()

#data_list=glob.glob("data/iwslt/"+args.src+"-"+args.tgt+"/*"+args.src+"-"+args.tgt+"."+args.src)

_abbreviations = [(re.compile('%s' % x[0], re.IGNORECASE), x[1]) for x in [
  ("&amp;", "&"),
  ("quat;", ";"),
  ("♬", " "),
  ("<", " "),
  (">", " "),
  (":", " "),
  ("/", " "),
  ("_", " "),
  (";", " "),
  ("!", " "),
  ("\[", " "),
  ("]", " "),
  ("\)", " "),
  ("\(", " "),
  ("\+", " "),
  ("=", " "),
  ("@", " "),
  ("#", " "),
  
  ("\^", " "),
  ("\*", " "),
  ("`", " "),
  
  ("\"", " "),
  ("-", " "),
  ("\.", " "),
  (",", " "),
]]

def expand_abbreviations(text):
    #import pdb; pdb.set_trace()    
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    text= del_space(text)
    text=re.sub('^\'', '', text)
    text=re.sub('\'$', '', text)
    return text
    
def del_space(text):
    while "  " in text:
        text=text.replace("  "," ")
    while " \'" in text:
        text=text.replace(" \'"," ")
    while "\' " in text:
        text=text.replace("\' "," ")
    return text.replace("?", " ?")

def clean_en(text):
    text_ = text.strip().replace("\\"," ")#.split("@")
    text=clean_text(text_,["english_cleaners"])
    text=expand_abbreviations(text)
    text=re.sub("&", " and ",text)
    text=re.sub("%", " percent ",text)
    text=re.sub("\$", " dollar ",text)
    text=del_space(text)
    text=re.sub('^ ', '', text)
    text=re.sub(' $', '', text)
    return text

def clean_other(text):
    text_ = text.strip().replace("\\"," ")#.split("@")
    text=clean_text(text_,["transliteration_cleaners"])
    text=expand_abbreviations(text)
    text=del_space(text)
    text=re.sub('^ ', '', text)
    text=re.sub(' $', '', text)
    return text


for i in codecs.open(args.src):
    key,value=i.split("@")
    value=clean_en(value)
    print(key+"@"+" ".join(list(value.replace(" ","_"))))

"""
label={}
en=codecs.open("data/root.en").readlines()
for idx,value in enumerate(en):
    key="TED-NMT-"+str(idx).zfill(len(list(str(len(en)))))
    label[value.strip()]=key

label_sp={}
for i in codecs.open("data/root_sp.en"):
    key,value=i.strip().split("@")
    label_sp[value]=key

for d in data_list:
    _name=os.path.basename(d).replace("."+args.src+"-"+args.tgt,"").replace("."+args.src,"")
    _dir=os.path.dirname(d)
    f=os.path.join(_dir,_name+"."+args.src+"-"+args.tgt+".")
    out=codecs.open(os.path.join(_dir,_name+"."+args.src+"-"+args.tgt),"w")
    for src,tgt in zip(codecs.open(f+args.src),codecs.open(f+args.tgt)):
        src=clean_en(src)
        tgt=clean_other(tgt) if args.tgt != "ja" else tgt.strip()
        if src in label and len(tgt)>1:
            out.write(label[src]+"@"+src+"@"+tgt+"\n")
        else:
            print(src+"@"+tgt)
            
        if src in label_sp and len(tgt)>1:
            out.write(label_sp[src]+"@"+src+"@"+tgt+"\n")
"""