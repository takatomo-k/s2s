"""Module defining inputters.

Inputters implement the logic of transforming raw data to vectorized inputs,
e.g., from a line of text to a sequence of embeddings.
"""
import os
from onmt.inputters.inputter import \
     get_fields, OrderedIterator, \
     filter_example
from onmt.inputters.dataset_base import Dataset
from onmt.inputters.text_dataset import text_sort_key, TextDataReader, TextMultiField
from onmt.inputters.image_dataset import img_sort_key, ImageDataReader
from onmt.inputters.audio_dataset import audio_sort_key, AudioDataReader
from onmt.inputters.mel_dataset import mel_sort_key, MelDataReader, MelSeqField

from onmt.inputters.datareader_base import DataReaderBase

str2sortkey = {
    'word': text_sort_key, 'bpe': text_sort_key, 'char': text_sort_key,'text': text_sort_key,
    'img': img_sort_key, 'audio': audio_sort_key, 'mel': mel_sort_key
    }

def reader(opt):
    #import pdb; pdb.set_trace()
    readers=[]
    for i in opt.train:
        if "text" in i:
            readers.append(TextDataReader.from_opt(opt))
        elif "mel" in i:
            readers.append(MelDataReader.from_opt(opt))
        elif "audio" in i:
            readers.append(AudioDataReader.from_opt(opt))
    return readers

def sortkey(opt):
    return str2sortkey[opt.data_type]


__all__ = ['Dataset', 'get_fields', 'DataReaderBase',
           'filter_example',
           'OrderedIterator',
           'text_sort_key', 'img_sort_key', 'audio_sort_key','mel_sort_key',
           'TextDataReader', 'ImageDataReader', 'AudioDataReader', 'MelDataReader',
           'TextMultiField','MelSeqField']
