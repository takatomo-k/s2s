"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import re
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import os
import onmt.inputters as inputters

from onmt.models import str2model
from onmt.encoders import str2enc

from onmt.decoders import str2dec

from onmt.modules import *
from onmt.modules.tts_modules import LinearGenerator
from onmt.utils.logging import logger
from onmt.utils.parse import ArgumentParser
from onmt.inputters import TextTransform, AudioTransform
from onmt.generators import str2gen



def build_embeddings(opt, fields):
    """
    Args:
        opt: the option in current environment.
        text_field(TextMultiField): word and feats field.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    def text_emb(opt, vocab):
        pad_indices = [vocab.pad_idx]
        word_padding_idx = pad_indices[0]
        num_embs = [len(vocab)]
        num_word_embeddings, num_feat_embeddings = num_embs[0], num_embs[1:]
        emb = Embeddings(
            word_vec_size=opt.src_word_vec_size,
            feat_merge=opt.feat_merge,
            feat_vec_exponent=opt.feat_vec_exponent,
            feat_vec_size=opt.feat_vec_size,
            dropout=opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            word_padding_idx=word_padding_idx,
            word_vocab_size=num_word_embeddings,
            sparse=opt.optim == "sparseadam",
            fix_word_vecs=opt.fix_word_vecs_enc
        )
        return emb
    
    if isinstance(fields["src"], TextTransform):
        src_emb= text_emb(opt, fields["src"])

    if isinstance(fields["tgt"], TextTransform):
        tgt_emb= text_emb(opt, fields["tgt"])
    
    # Share the embedding matrix - preprocess with share_vocab required.
    if opt.model_type == 'nmt':
        if opt.share_emb and len(fields["src"]) == len(fields["tgt"]):
            tgt_emb.word_lut.weight = src_emb.word_lut.weight
    elif opt.model_type == 'tts':
        src_emb = PrenetConv(src_emb, opt.tgt_word_vec_size,
                  opt.rnn_size, opt.dropout[0] if isinstance(opt.dropout,list) else opt.dropout
                )
        
        tgt_emb = Prenet(len(fields["tgt"]),opt.rnn_size*2, 
                  opt.rnn_size, opt.dropout[0] if isinstance(opt.dropout,list) else opt.dropout
                )
    elif opt.model_type in {'asr'}:
        src_emb = Prenet(len(fields["src"]),opt.rnn_size*2,
                  opt.rnn_size, opt.dropout[0] if type(opt.dropout) is list else opt.dropout
                )
        src_emb = PrenetConv(src_emb,opt.rnn_size,opt.rnn_size,opt.dropout)
    
    elif opt.model_type in {'vocoder'}:
        src_emb = None
        tgt_emb = None

    return src_emb, tgt_emb

def build_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    if opt.model_type == 'vocoder':
        return PostConvNet(80, 80) #post_mel generater

    return str2enc(opt.encoder_type).from_opt(opt, embeddings)

def build_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    if opt.model_type == 'vocoder':
        return  MagGenerater(opt.rnn_size) # mag generater

    dec_type = "ifrnn" if opt.decoder_type == "rnn" and opt.input_feed \
               else opt.decoder_type
    return str2dec(dec_type).from_opt(opt, embeddings)

def build_generator(opt, fields, model):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    return str2gen[opt.model_type](opt, fields["tgt"], model.decoder)
    
def _build_multi_task_model(pre_model, post_model, gpu, checkpoint=None, gpu_id=None):
    assert pre_model is not None or post_model is not None
    if gpu and gpu_id is not None:
        device = torch.device("cuda", gpu_id)
    elif gpu and not gpu_id:
        device = torch.device("cuda")
    elif not gpu:
        device = torch.device("cpu")
    
    generator = post_model.generator
    if checkpoint is not None:
        # end of patch for backward compatibility
        model.load_state_dict(checkpoint['model'], strict=False)
        generator.load_state_dict(checkpoint['generator'], strict=False)
    model.generator = generator
    model.to(device)
    
    return model

def load_model(model,generator,checkpoint,model_opt):
    
    if checkpoint is not None:# and not isinstance(checkpoint['vocab']['tgt'],MelTransform):
        # This preserves backward-compat for models using customed layernorm
        def fix_key(s):
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                       r'\1.layer_norm\2.bias', s)
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                       r'\1.layer_norm\2.weight', s)
            return s

        checkpoint['model'] = {fix_key(k): v
                               for k, v in checkpoint['model'].items()}
        # end of patch for backward compatibility
        try:
            model.load_state_dict(checkpoint['model'], strict=False)
        except:
            pass
        try:
            generator.load_state_dict(checkpoint['generator'], strict=False)
        except:
            pass

    model.generator = generator
    
    return model
    
def build_base_model(model_opt, fields, checkpoint=None):
    if checkpoint is not None:
        model_opt=checkpoint['opt']
        fileds=checkpoint['vocab']

    src_emb, tgt_emb = build_embeddings(model_opt, fields)

    # Build encoder.
    encoder = build_encoder(model_opt, src_emb) 

    # Build decoder.
    decoder = build_decoder(model_opt, tgt_emb) 
    
    # Build NMTModel(= encoder + decoder).
    model = str2model(model_opt.model_type)(encoder, decoder)

    # Build Generator.
    
    generator = build_generator(model_opt, fields, model) 
    
    if 'spkr' in fields:
        setattr(model,'spkr_emb',nn.Embedding(len(fields['spkr'],400,bias=True)))
    
    model=load_model(model,generator,checkpoint,model_opt)
    #model.update_dropout(0.4)
    return model


def build_model(model_opt, opt, fields, device,checkpoint):
    logger.info('Building single model...')
    model = build_base_model(model_opt, fields, checkpoint)
    
    logger.info(model)
    model.to(device)
    if model_opt.model_dtype == 'fp16':
        model.half()
    
    #model.decoder.noise.sigma=opt.noisy_dec
    model.update_dropout(opt.dropout[0] if isinstance(opt.dropout,list) else opt.dropout)
    return model

def build_multi_model(model_opt, opt, fields, device, checkpoint):
    #
    models=[]
    for path in opt.pretrain:
        #import pdb; pdb.set_trace()
        _check=torch.load(path)
        #import pdb; pdb.set_trace()
        if _check['opt'].model_type in {"single","vocoder","nmt","tts","asr"}:
            models.append(build_base_model(_check['opt'],_check['vocab'],_check))
        else:
            models.append(build_multi_model(_check['opt'],_check['opt'],_check['vocab'],device,_check))
    model = str2model(opt.model_type)(models)
    #import pdb; pdb.set_trace()
    generator = models[-1].generator 
    if opt.spkr:
        setattr(model,'spkr_emb',nn.Embedding(len(fields['spkr']),400))
    model=load_model(model, generator, checkpoint, model_opt)
    #if 
        
    model.update_dropout(opt.dropout[0] if isinstance(opt.dropout,list) else opt.dropout)
    
    #logger.info('Building ',model.__class__.__name__)
    logger.info(model)    
    model.to(device)
    if model_opt.model_dtype == 'fp16':
        model.half()
    return model

