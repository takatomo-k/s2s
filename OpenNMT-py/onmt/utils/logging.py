# -*- coding: utf-8 -*-
from __future__ import absolute_import
from datetime import datetime
import logging
import os

logger = logging.getLogger()


def make_log_path(opt):
    
    dropout="dropout"+str(opt.dropout[0]) if isinstance(opt.dropout,list) else "dropout"+str(opt.dropout)
    base_path = os.path.join(opt.save_model,opt.data.replace("data/ext/","model/").replace("exp/data/","model/").replace('nakamura-lab08','nakamura-lab05'),opt.model_type, opt.encoder_type,
                             "layers"+str(opt.layers),
                             dropout,str(opt.rnn_size))
    
    if opt.noisy_enc>0:
        base_path = os.path.join(base_path,
                     "noisyEnc"+str(opt.noisy_enc))
    if opt.noisy_dec>0:
        base_path = os.path.join(base_path,
                     "noisyDec"+str(opt.noisy_dec))
    if opt.share_dec_emb:
        base_path = os.path.join(base_path,"tied")
    if opt.gen_type > 0:
        base_path = os.path.join(base_path,
                     "gen_type"+str(opt.gen_type))

    #base_path += datetime.now().strftime("/%b-%d/%H-%M-%S/")
    return base_path
    

def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger
