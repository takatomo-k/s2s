'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

import re
from unidecode import unidecode
from onmt.utils.my_numbers import normalize_numbers


# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]


def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text


def expand_numbers(text):
  return normalize_numbers(text)


def lowercase(text):
  return text.lower()


def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
  return unidecode(text)


def basic_cleaners(text):
  '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def transliteration_cleaners(text):
  '''Pipeline for non-English text that transliterates to ASCII.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = text.replace(","," ").replace("."," ").replace("-"," ").replace("?"," ?")
  text = collapse_whitespace(text)
  return text

def spanish_cleaners(text):
  '''Pipeline for non-English text that transliterates to ASCII.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = text.replace("\""," ")
  text = collapse_whitespace(text)
  return text

def korean_cleaners(text):
  '''Pipeline for non-English text that transliterates to ASCII.'''
  #text = convert_to_ascii(text)
  #text = lowercase(text)
  text = collapse_whitespace(text).replace('.',' ').replace('?',' ?').replace(',',' ')
  return text


def english_cleaners(text):
  '''Pipeline for English text, including number and abbreviation expansion.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_numbers(text)
  text = expand_abbreviations(text)
  text = text.replace("\""," ").replace(","," ").replace("."," ").replace("-"," ").replace("?"," ?").replace(';',' ').replace(':',' ').replace('!','')
  text = collapse_whitespace(text)
  return text


import MeCab
import codecs
import argparse
from pykakasi import kakasi
import re
re_hiragana = re.compile(r'^[あ-ん]+$')
re_katakana = re.compile(r'[\u30A1-\u30F4]+')
re_kanji = re.compile(r'^[\u4E00-\u9FD0]+$')
j2h = kakasi()
j2h.setMode('J', 'H')  # J(Kanji) to H(Hiragana)
conv = j2h.getConverter()

k2h=kakasi()
k2h.setMode('K','H')
conv2=k2h.getConverter()
t = MeCab.Tagger('')



def japanese_cleaners(text):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    node=t.parse(text).replace("\t",",").split("\n")
    res=[]
    for i in node:
        #import pdb; pdb.set_trace()
        node=i
        if node.split(",")[0]!="*":
            res.append(conv2.do(node.split(",")[0]))
  
    text= " ".join(res[:-2])
    
    #print(text)
    return text

def hiragana_cleaners(text):
    return text

str2cleaners={'ja':japanese_cleaners,'en':english_cleaners,'ko':korean_cleaners,'es':spanish_cleaners,'hira':hiragana_cleaners}
