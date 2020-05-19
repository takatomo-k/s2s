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
import torch
from functools import partial
from collections import Counter, defaultdict
from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import split_corpus
import onmt.my_inputters as inputters
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
#from onmt.my_inputters.inputter import get_fields, get_base_name

def check_existing_pt_files(opt):
    """ Check if there are existing .pt files to avoid overwriting them """
    pattern = opt.save_data + '.{}*.pt'
    for t in ['train', 'valid']:
        path = pattern.format(t)
        if glob.glob(path):
            sys.stderr.write("Please backup existing pt files: %s, "
                             "to avoid overwriting them!\n" % path)


def build_save_dataset(opt, fields):
    for corpus_type, data in zip(['train', 'valid', 'test'],[opt.train,opt.valid,opt.test]):
        for key, value in fields.items():
            logger.info("%s features is %d"%(key,len(value)))
        #import pdb;pdb.set_trace()
        #logger.info("Reading source and target files: %s %s." % (data_list[0], data_list[-1]))
        #import pdb;pdb.set_trace()
        dataset = inputters.MyDataset(
            load_data(opt,data),
            fields,
            opt
        )
        if not os.path.exists(opt.save_data):
            os.makedirs(opt.save_data,exist_ok=True)
        shard_base = corpus_type
        data_path = "{:s}.{:s}.pt".\
            format(opt.save_data, shard_base)
        
        logger.info(" * saving %s data shard to %s."
                    % (shard_base, data_path))
        torch.save(dataset, data_path)

        del dataset
        gc.collect()
    
def main(opt):
    ArgumentParser.validate_preprocess_args(opt)
    torch.manual_seed(opt.seed)
    check_existing_pt_files(opt)

    init_logger(opt.log_file)
    if not os.path.exists(os.path.dirname(opt.save_data)):
        os.makedirs(os.path.dirname(opt.save_data))
        logger.info("Creating dirs..."+os.path.dirname(opt.save_data))
    logger.info("Extracting features...")
    fields = get_fields(opt)
    torch.save(fields, opt.save_data + '.vocab.pt')
    logger.info("Building & saving training data...")
    build_save_dataset(opt, fields
    )

def fields_getters(key, path, opt, domain):
    if domain in {"char","sub","word"}:
        return inputters.TextTransform(key, path, opt.src_words_min_frequency if key=='src' else opt.tgt_words_min_frequency)
    elif domain == "mfcc" :
        return inputters.AudioTransform(key, path)
    elif domain in {"mag","mel" }:
        return inputters.MelTransform(key, path)
    elif domain=='spkr':
        return inputters.SpeakerTransform(key, path)
    elif domain=='advr':
        return inputters.AdversarialTransform(key)
    else:
        import pdb; pdb.set_trace()

def get_fields(opt):
    fields={}
    #import pdb;pdb.set_trace()
    for n,l,d in zip(opt.name, opt.lang, opt.domain):
        dataset = []
        for path in opt.train:
            #if os.path.exists(os.path.join("/home/is/takatomo-k/data/", l, path, d)):
            dataset.extend([d for d in codecs.open(os.path.join("/home/is/takatomo-k/data/", l, path, d),"r")])
                    
        fields[n]=fields_getters(n,dataset,opt,d)
    torch.save(fields, opt.save_data + '.vocab.pt')
    
    return fields


def load_data(opt, data):
    dataset={}
    for n,l,d in zip(opt.name, opt.lang, opt.domain):
        dataset[n] = []
        for path in data:
            #if os.path.exists(os.path.join("/home/is/takatomo-k/data/", l, path, d)):
            dataset[n].extend([d for d in codecs.open(os.path.join("/home/is/takatomo-k/data/", l, path, d),"r")])
    
    return dataset

def _get_parser():
    parser = ArgumentParser(description='preprocess.py')
    opts.config_opts(parser)
    opts.preprocess_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    for idx in range(len(opt.train)):
        opt.train[idx]+="/train"
    for idx in range(len(opt.valid)):
        opt.valid[idx]+="/valid"
    for idx in range(len(opt.test)):
        opt.test[idx]+="/test"
    main(opt)
