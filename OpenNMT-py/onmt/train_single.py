#!/usr/bin/env python
"""Training on a single process."""
import os

import torch
from torch.utils.data import DataLoader
from onmt.inputters.inputter import MySampler, collate_fn
from onmt.model_builder import build_model,build_multi_model
from onmt.utils.optimizers import Optimizer
from onmt.utils.misc import set_random_seed
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser
from onmt.utils.loss import AdversarialLossCompute
from onmt.generators import SpeakerGenerator, AdversarialGenerator
from onmt.utils.misc import use_gpu

def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)

def _tally_parameters(model):
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        else:
            dec += param.nelement()
    return enc + dec, enc, dec


def configure_process(opt, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(opt.seed, device_id >= 0)

def load_pre_train(path):
    logger.info('Loading pre-train model from %s' % path)
    checkpoint = torch.load(path,
                            map_location=lambda storage, loc: storage)

    opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
    model_opt=opt
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
    fields = checkpoint['vocab']
    model = build_model(model_opt, opt, fields, checkpoint)
    return model

def load_dataset(path):
    data=torch.load(path+"/train.pt")
    natural=0
    generate=0
    datasets={k:v for k, v in data.examples['src'].examples.items()}
    for k,v in datasets.items():
        if 'natural' in v :
            natural+=1
        else:
            generate+=1
    print("natural",natural,"generate",generate)
    
    return torch.load(path+"/train.pt"), torch.load(path+"/valid.pt"), torch.load(path+"/test.pt") 
def adversarial(is_adversarial, model, opt,field):
    if is_adversarial:
        field=field['mid']
        if isinstance(field,SpeakerTransform):
            adversarial = SpeakerGenerator.from_opt(opt,field)
            criterion = torch.nn.NLLLoss()
        else:
            adversarial = AdversarialGenerator.from_opt(opt,field)
            criterion = torch.nn.BCELoss()
        adversarial.cuda()
        loss_ad = AdversarialLossCompute(adversarial,criterion)
        optim_ad = Optimizer.from_opt(adversarial, opt)
        return optim_ad, loss_ad
    else:
        return None,None

def main(opt, device_id):
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.
    configure_process(opt, device_id)
    init_logger(opt.log_file)
    assert len(opt.accum_count) == len(opt.accum_steps), \
        'Number of accum_count values must match number of accum_steps'
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)

        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
    else:
        checkpoint = None
        model_opt = opt
    vocab = torch.load(opt.data + '/vocab.pt')

    if opt.train_from and not opt.reset_optim in {'score','all'}:
        valid_loss = checkpoint['valid_loss']
        test_score = checkpoint['test_score']
    else:
        valid_loss = 100000
        test_score = -1

    fields = vocab
    # Report src and tgt vocab sizes, including for features
    for idx in fields.keys():
        logger.info(' * %s feat size = %d' % (idx, len(fields[idx])))

    # Detect device
    gpu=use_gpu(opt)
    if gpu :
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # Build model.
    if opt.model_type in {'single','nmt','tts','asr','vocoder'}:
        model = build_model(model_opt, opt, fields, device, checkpoint)
        n_params, enc, dec = _tally_parameters(model)
        logger.info('encoder: %d' % enc)
        logger.info('decoder: %d' % dec)
        logger.info('* number of parameters: %d' % n_params)
    else:
        model = build_multi_model(model_opt, opt, fields, device, checkpoint)
    
    #_check_save_model_path(opt)
    # Build optimizer.
    optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)
    #if opt.adversarial:
    optim_ad, loss_ad = adversarial(opt.adversarial, model, model_opt, vocab)
    trainer = build_trainer(fields, opt, device_id, model, optim,
        model_saver=model_saver,optim_ad=optim_ad,loss_ad=loss_ad)

    train_set, valid_set, test_set =load_dataset(opt.data)
    
    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
    train_steps = opt.train_steps
    if opt.single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0
    trainer.train(
        train_set,
        opt.batch_size,
        train_steps,
        save_checkpoint_steps=opt.save_checkpoint_steps,
        valid_set=valid_set,
        valid_batch_size=min(opt.valid_batch_size,opt.batch_size),
        valid_steps=opt.valid_steps,
        test_set=test_set,
        valid_loss=valid_loss,
        test_score=test_score
        )

    trainer.report_manager.tensorboard_writer.close()
