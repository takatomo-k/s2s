#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data / features files and build vocabulary
"""
import codecs
import glob
import sys
import gc
import os
import glob
import torch
from functools import partial
from collections import Counter, defaultdict
from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import split_corpus
import onmt.inputters as inputters
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
import random
import copy

def check_existing_pt_files(opt,corpus_type='train'):
    out_dir=os.path.join(out,opt.lang[0]+"2"+opt.lang[-1]+"/","2".join(opt.domain))
    out_dir=out_dir.replace(corpus_type,'').replace('.','')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir

def build_save_dataset(opt, fields):
    for corpus_type in ['train','test']:
        for key, value in fields.items():
            logger.info("%s features is %d"%(key,len(value)))
        dataset = inputters.MyDataset(
            load_data(opt,corpus_type),
            fields,
            opt,
            training= corpus_type=='train'
        )
        out_dir=check_existing_pt_files(opt,corpus_type)
        shard_base = corpus_type
        data_path = "{:s}/{:s}.pt".\
            format(out_dir, shard_base)
        
        logger.info(" * saving %s data shard to %s."
                    % (shard_base, data_path))

        if corpus_type=='train':
            valid=copy.deepcopy(dataset)
            data=[i for i in dataset.keys()]
            random.shuffle(data)
            dataset.remove(data[:1000])
            valid.remove(data[1000:])
            #import pdb; pdb.set_trace()
            torch.save(valid,data_path.replace('train.pt','valid.pt'))            
            torch.save(fields, data_path.replace('train.pt','vocab.pt'))
        torch.save(dataset, data_path)
        
        del dataset
        gc.collect()

def main(opt):
    ArgumentParser.validate_preprocess_args(opt)
    torch.manual_seed(opt.seed)
    #check_existing_pt_files(opt)

    init_logger(opt.log_file)
    #if not os.path.exists(os.path.dirname(opt.save_data)):
    #    os.makedirs(os.path.dirname(opt.save_data))
    #    logger.info("Creating dirs..."+os.path.dirname(opt.save_data))
    logger.info("Extracting features...")
    fields = get_fields(opt)
    logger.info("Building & saving training data...")
    build_save_dataset(opt, fields)

def fields_getters(key, data, opt, domain, lang):
    if domain in {"chr","sub","wrd"}:
        return inputters.TextTransform(key, data, domain, lang)
    else :
        return inputters.AudioTransform(key, data, domain)

def get_fields(opt):
    fields={}
    #import pdb; pdb.set_trace()
    dataset= load_data(opt, 'train')
    for n,l,d in zip(opt.name, opt.lang, opt.domain):
        fields[n]=fields_getters(n, dataset[n], opt, d, l)
    
    #torch.save(fields, opt.save_data + '.vocab.pt')
    
    return fields


def load_data(opt, data):
    dataset={}
    for n,l,d in zip(opt.name, opt.lang, opt.domain):
        dataset[n] = []
        path=os.path.join('data/btec/',l+'.'+data)
        if os.path.exists(path):
            for idx,i in enumerate(codecs.open(path,"r",encoding='utf8')):
                dataset[n].append(i.strip())
    return dataset

def _get_parser():
    parser = ArgumentParser(description='preprocess.py')
    opts.config_opts(parser)
    opts.preprocess_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
