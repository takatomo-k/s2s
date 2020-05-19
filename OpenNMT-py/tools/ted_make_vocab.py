import sys
import codecs
import unicodedata

lines= codecs.open(sys.argv[1],encoding="utf8").readlines()
def is_digit(num):
    for i in ["0","1","2","3","4","5","6","7","8","9"]:
        if i in num:
            return True
        else:
            return False

def fix_digit(sen):
    ret=[]
    last=False
    for i in sen.split():
        if is_digit(i):
            if last:
                i=ret[-1]+i
                del ret[-1]
            if len(i)>4:
                #import pdb;pdb.set_trace()
                i=i.replace(".",",").replace(",","")
            last=True
        else:
            last=False
        ret.append(i)
    return " ".join(ret)

def read(form="NFC"):
    en_=[]
    fr_=[]
    for i in lines:
        i=i.strip()
        try:
            en,fr =i.split("|")
            en_.append(unicodedata.normalize(form,en.lower()).replace("%"," percent ").replace("&"," and ").replace("$"," dollars "))
            fr_.append(unicodedata.normalize(form,fr.lower()).replace("%"," pour cent ").replace("&"," et ").replace("$"," dollars "))
        except:
            #import pdb;pdb.set_trace()
            continue
    return en_, fr_

def make_dict(en_, fr_):
    en_dict={}
    fr_dict={}
    for en,fr in zip(en_,fr_):
        for e in list(en):
            en_dict[e]=1 if e not in en_dict else en_dict[e]+1
        for f in list(fr):
            fr_dict[f]=1 if f not in fr_dict else fr_dict[f]+1
    print(len(en_dict),len(fr_dict))
    return en_dict,fr_dict

def write_file(en_,fr_,en_dict,fr_dict,count):
    out_txt=codecs.open("ted_all","w",encoding="utf8")
    out_gomi=codecs.open("ted_gomi","w",encoding="utf8")
    for en,fr in zip(en_,fr_):
        txt=[]
        gomi=[]
        en=fix_digit(en)
        fr=fix_digit(fr)
        for e in list(en):
            try:
                if en_dict[e] >count:
                    txt.append(e)
                else:
                    #import pdb;pdb.set_trace()
                    gomi.append(e)
            except:
                pass
                #import pdb;pdb.set_trace()
        txt.append("|")
        gomi.append("|")
        for e in list(fr):
            
            if fr_dict[e] >count:
                txt.append(e)
            else:
                #import pdb;pdb.set_trace()
                gomi.append(e)
        txt.append("\n")
        gomi.append("\n")
        out_txt.write("".join(txt))
        out_gomi.write("".join(gomi))


en,fr=read("NFKD")
en_dict,fr_dict = make_dict(en,fr)
out_fr=codecs.open("fr.dict","w",encoding="utf8")
out_en=codecs.open("en.dict","w",encoding="utf8")
for (k,v) in sorted(fr_dict.items(), key=lambda x:x[1]):
    out_fr.write(str(v)+" "+str(k)+"\n")

for (k,v) in sorted(en_dict.items(), key=lambda x:x[1]):
    out_en.write(str(v)+" "+str(k)+"\n")
out_fr.close()
out_en.close()
write_file(en,fr,en_dict,fr_dict,400)
"""
for i in lines:
    i=i.strip()
    try:
        en,fr =i.split("|")
        en=unicodedata.normalize("NFC",en.lower())
        fr=unicodedata.normalize("NFC",fr.lower())
    except:
        #import pdb;pdb.set_trace()
        continue
    txt=[]
    gomi=[]
    for e in list(en):
        if en_dict[e] >200:
            txt.append(e)
        else:
            gomi.append(e)
    txt.append("|")
    gomi.append("|")

    for e in list(fr):
        if fr_dict[e] >200:
            txt.append(e)
        else:
            gomi.append(e)

    print("".join(gomi)+">"+"".join(txt))
"""