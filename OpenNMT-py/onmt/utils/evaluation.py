import torch
import Levenshtein
import warnings

from  nltk.translate.bleu_score import sentence_bleu
warnings.filterwarnings('ignore')

def bleu(ref, hyp,_type="word"):
    #if _type=="char":
    #    hyp=hyp.replace(" ","").replace("_"," ")
    #    ref=ref.replace(" ","").replace("_"," ")
    return sentence_bleu([hyp], ref)

def wer(ref, hyp,_type="word"):
    if len(ref)==0 or len(hyp)==0:
        return 0
    #if _type=="char":
    #    hyp=hyp.replace(" ","").replace("_"," ")
    #    ref=ref.replace(" ","").replace("_"," ")
    return Levenshtein.distance(hyp, ref)/len(ref)
