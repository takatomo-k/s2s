"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

from copy import deepcopy
import itertools
import torch
import traceback
import tqdm
import onmt.utils
from onmt.utils.logging import logger
#from onmt.utils.loss import TTSLossCompute
from torch.utils.data import DataLoader
from onmt.inputters.inputter import MySampler, collate_fn,Batch
from onmt.models import NMTModel, CascadeModel, NeuralCascadeModel, TranscoderModel, TranscoderModel2, GoogleMultitaskModel
from onmt.translate.translator import build_translator
#from multiprocessing_generator import ParallelGenerator
import numpy as np
def build_trainer(fields, opt, device_id, model,  optim, model_saver=None,optim_ad=None, loss_ad=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    train_loss = onmt.utils.loss.build_loss_compute(model, opt)
    valid_loss = onmt.utils.loss.build_loss_compute(model, opt, train=False)

    #trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches if opt.model_dtype == 'fp32' else 0
    norm_method = opt.normalization
    accum_count = opt.accum_count
    accum_steps = opt.accum_steps
    n_gpu = opt.world_size
    average_decay = opt.average_decay
    average_every = opt.average_every
    dropout = opt.dropout
    dropout_steps = opt.dropout_steps
    if device_id >= 0:
        gpu_rank = opt.gpu_ranks[device_id]
    else:
        gpu_rank = 0
        n_gpu = 0
    gpu_verbose_level = opt.gpu_verbose_level

    earlystopper = onmt.utils.EarlyStopping(
        opt.early_stopping, scorers=onmt.utils.scorers_from_opts(opt)) \
        if opt.early_stopping > 0 else None
    #import pdb; pdb.set_trace()
    report_manager = onmt.utils.build_report_manager(opt,fields)
    
    
    translator = build_translator(model, torch.load(opt.data+"vocab.pt"), opt, report_score=True, device_id=device_id)
    
    trainer = onmt.Trainer(model, translator, train_loss, valid_loss, optim,
                           shard_size, norm_method,
                           accum_count, accum_steps,
                           n_gpu, gpu_rank,
                           gpu_verbose_level, report_manager,
                           model_saver=model_saver if gpu_rank == 0 else None,
                           average_decay=average_decay,
                           average_every=average_every,
                           model_dtype=opt.model_dtype,
                           earlystopper=earlystopper,
                           dropout=dropout,
                           dropout_steps=dropout_steps,
                           device_id=device_id,
                           optim_ad=optim_ad,
                           loss_ad=loss_ad)
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            accum_count(list): accumulate gradients this many times.
            accum_steps(list): steps for accum gradients changes.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, translator, train_loss, valid_loss, optim,
                 shard_size=0,
                 norm_method="sents", accum_count=[1],
                 accum_steps=[0],
                 n_gpu=1, gpu_rank=1,
                 gpu_verbose_level=0, report_manager=None,
                 model_saver=None, average_decay=0,
                 average_every=1, model_dtype='fp32',
                 earlystopper=None, dropout=[0.3],
                 dropout_steps=[0], device_id = 0,
                 loss_ad = None, optim_ad=None):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.shard_size = 0
        self.norm_method = norm_method
        self.accum_count_l = accum_count
        self.accum_count = accum_count[0]
        self.accum_steps = accum_steps
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.model_saver = model_saver
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper
        self.dropout = dropout
        self.dropout_steps = dropout_steps
        self.translator = translator
        self.device_id = device_id
        self.loss_ad = loss_ad
        self.optim_ad = optim_ad

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0
            
        # Set model in training mode.
        self.model.train()
        
    def _accum_count(self, step):
        for i in range(len(self.accum_steps)):
            if step > self.accum_steps[i]:
                _accum = self.accum_count_l[i]
        return _accum

    def _maybe_update_dropout(self, step):
        for i in range(len(self.dropout_steps)):
            if step > 1 and step == self.dropout_steps[i] + 1:
                self.model.update_dropout(self.dropout[i])
                logger.info("Updated dropout to %f from step %d"
                            % (self.dropout[i], step))

    def _accum_batches(self, iterator):
        batches = []
        normalization = 0
        self.accum_count = self._accum_count(self.optim.training_step)
        for batch in iterator:
            batch.to(self.device_id)
            batches.append(batch)
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:, :, 0].ne(
                    self.train_loss.padding_idx).sum()
                normalization += num_tokens.item()
            else:
                normalization += batch.batch_size
            if len(batches) == self.accum_count:
                yield batches, normalization
                self.accum_count = self._accum_count(self.optim.training_step)
                batches = []
                normalization = 0
        if batches:
            yield batches, normalization

    def _update_average(self, step):
        if self.moving_average is None:
            copy_params = [params.detach().float()
                           for params in self.model.parameters()]
            self.moving_average = copy_params
        else:
            average_decay = max(self.average_decay,
                                1 - (step + 1)/(step + 10))
            for (i, avg), cpt in zip(enumerate(self.moving_average),
                                     self.model.parameters()):
                self.moving_average[i] = \
                    (1 - average_decay) * avg + \
                    cpt.detach().float() * average_decay
    
    def train(self, train_set, train_batch_size, train_steps,
              save_checkpoint_steps,
              valid_set, valid_batch_size, valid_steps,
              test_set, valid_loss, test_score):
        """
        The main training loop by iterating over `train_iter` and possibly
        running validation on `valid_iter`.

        Args:
            train_iter: A generator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.

        Returns:
            The gathered statistics.
        """
        total_stats = onmt.utils.Statistics()
        report_stats = onmt.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        step = self.optim.training_step
        #import pdb; pdb.set_trace()
        while  step < train_steps:
            train_iter = DataLoader(train_set, batch_size=train_batch_size,sampler=MySampler(train_set,train_batch_size,is_train=True,step=step),
                    collate_fn=collate_fn, num_workers=32,shuffle=False)
            if self.n_gpu > 1:
                train_iter = itertools.islice(train_iter, self.gpu_rank, None, self.n_gpu)
            epoch = int(step/len(train_iter))
            logger.info("Training epoch: %d"
                                % (epoch))
            for i, (batches, normalization) in enumerate(
                    self._accum_batches(train_iter)):
                step = self.optim.training_step
                
                # UPDATE DROPOUT
                self._maybe_update_dropout(step)
                #print(step)
                if self.gpu_verbose_level > 1:
                    logger.info("GpuRank %d: index: %d", self.gpu_rank, i)
                if self.gpu_verbose_level > 0:
                    logger.info("GpuRank %d: reduce_counter: %d \
                                n_minibatch %d"
                                % (self.gpu_rank, i + 1, len(batches)))
                if self.n_gpu > 1:
                    normalization = sum(onmt.utils.distributed
                                        .all_gather_list
                                        (normalization))

                attns = self._gradient_accumulation(
                        batches, normalization, total_stats,
                    report_stats)                

                if self.average_decay > 0 and i % self.average_every == 0:
                    self._update_average(step)

                report_stats = self._maybe_report_training(
                    step, train_steps,
                    self.optim.learning_rate(),
                    report_stats, attns)
                
                if step % save_checkpoint_steps == 0:
                    self.model_saver.save(step, "train", valid_loss, 
                                        test_score, 
                                        moving_average=self.moving_average)
                if valid_set is not None and step % valid_steps == 0:
                    valid_iter = DataLoader(valid_set, batch_size=valid_batch_size, sampler=MySampler(valid_set,valid_batch_size, is_train=True),
                    collate_fn=collate_fn, num_workers=16)
                    if self.gpu_verbose_level > 0:
                        logger.info('GpuRank %d: validate step %d'
                                    % (self.gpu_rank, step))
                    
                    _valid_loss, valid_stats, attns = self.validate(
                        valid_iter, moving_average=self.moving_average)
                    if self.gpu_verbose_level > 0:
                        logger.info('GpuRank %d: gather valid stat \
                                    step %d' % (self.gpu_rank, step))
                    valid_stats = self._maybe_gather_stats(valid_stats)
                    if self.gpu_verbose_level > 0:
                        logger.info('GpuRank %d: report stat step %d'
                                    % (self.gpu_rank, step))
                    self._report_step(self.optim.learning_rate(),
                                    step, valid_stats=valid_stats, attns=attns)
                    #import pdb; pdb.set_trace()
                    if True:#valid_loss > _valid_loss:
                        logger.info('Update valid loss %f -> %f' % (valid_loss, _valid_loss))
                        valid_loss = _valid_loss
                        #self.model_saver.save(step, "valid", valid_loss, 
                        #                    test_score, 
                        #                    moving_average=self.moving_average)
                        #import pdb; pdb.set_trace()
                        
                        if self.translator is not None:
                            test_stats, attns = self.test(
                            test_set, moving_average=self.moving_average)
                            #import pdb; pdb.set_trace()
                            bleu_score = self._maybe_report_test(step, test_stats, attns)
                            if self.gpu_verbose_level > 0:
                                logger.info('GpuRank %d: gather test stat \
                                            step %d' % (self.gpu_rank, step))
                            if test_score < bleu_score:
                                logger.info('Update test score %f -> %f'
                                            % (test_score, bleu_score))
                                test_score = bleu_score
                                self.model_saver.save(step, "test", valid_loss, 
                                                    test_score, 
                                                    moving_average=self.moving_average)
                            else:
                                logger.info('Best test score %f Current test score %f'
                                            % (test_score, bleu_score))
                        
                    else:
                        logger.info('Best valid loss %f Current valid loss %f'
                                    % (valid_loss, _valid_loss))
                    # Run patience mechanism
                    if self.earlystopper is not None:
                        self.earlystopper(valid_stats, step)
                        # If the patience has reached the limit, stop training
                        if self.earlystopper.has_stopped():
                            break
                                    
                

            if train_steps > 0 and step >= train_steps:
                break

        self.model_saver.save(step, "train", valid_loss, 
                              test_score, 
                              moving_average=self.moving_average)
        
        return total_stats, attns

    def validate(self, valid_iter, moving_average=None):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        if moving_average:
            valid_model = deepcopy(self.model)
            for avg, param in zip(self.moving_average,
                                  valid_model.parameters()):
                param.data = avg.data.half() if self.model_dtype == "fp16" \
                    else avg.data
        else:
            valid_model = self.model

        # Set model in validating mode.
        valid_model.eval()
        valid_loss=0
        with torch.no_grad():
            stats = onmt.utils.Statistics()
            for batch in valid_iter:
                batch.to(self.device_id)
                # F-prop through the model.
                result = valid_model(batch=batch)

                # Compute loss.
                loss, batch_stats, pred = self.valid_loss(batch, result)
                # Update statistics.
                #stats.update(batch_stats)
                valid_loss+=loss.item()
        try:
            result['pred']= torch.tensor(pred)[1:-1,0].detach().cpu().numpy()#batch.tgt[0].detach().cpu().numpy()
            result['tgt']=batch.tgt[0][1:-1,0].detach().cpu().numpy()
            result['src']=batch.src[0][:,0].detach().cpu().numpy()
        except:
            pass

        if moving_average:
            del valid_model
        else:
            # Set model back to training mode.
            valid_model.train()
        return valid_loss/max(len(valid_iter),1),stats, result

    def test(self, test_set, moving_average=None):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        
        self.model.eval()
        stats,attns=[],[]
        with torch.no_grad():
            data_iter = DataLoader(test_set, batch_size=1, sampler=MySampler(test_set, 1, is_train=False),
                    collate_fn=collate_fn, num_workers=16)
            cnt=0
            for batch in tqdm.tqdm(data_iter, total=100):
                #import pdb; pdb.set_trace()
                if isinstance(self.model,NMTModel) or isinstance(self.model,GoogleMultitaskModel):
                    stat, attn, context,probs = self.translator['tgt'].translate([batch])
                    stat=stat

                    attn={'pred':attn[0]}
                    #import pdb; pdb.set_trace()
                    if 'pred_wld' in stat:
                        attn['tgt']= stat['pred_wld'][0][:,0].detach().cpu().numpy()
                        attn['tgt']= stat['ref_wld'][0][:,0].detach().cpu().numpy()

                elif isinstance(self.model,TranscoderModel2) or isinstance(self.model,TranscoderModel):
                    #import pdb; pdb.set_trace()
                    stat={}
                    attn={}
                    my_batch=Batch()
                    setattr(my_batch, "src", (batch.src))
                    setattr(my_batch, "tgt", (batch.src_txt))
                    setattr(my_batch, 'batch_size', batch.batch_size)
                    setattr(my_batch, 'indices', batch.indices)
                    
                    _stat, _attn, asr_context,probs = self.translator['src_txt'].translate([my_batch], model=self.model.asr)
                    stat['asr_ref'],stat['asr_pred']=_stat[0]['tgt'],_stat[0]['pred']
                    attn['asr_attn']=_attn[0]
                    #import pdb; pdb.set_trace()
                    asr_context=asr_context[0][0].unsqueeze(1)[:-1]
                    setattr(my_batch, "src", (asr_context,torch.tensor([asr_context.size(0)])))
                    setattr(my_batch, "tgt", (batch.tgt_txt))
                    #import pdb; pdb.set_trace()
                    _stat, _attn, nmt_context, probs = self.translator['tgt_txt'].translate([my_batch], model=self.model.nmt)
                    stat['nmt_ref'],stat['nmt_pred']=_stat[0]['tgt'],_stat[0]['pred']
                    attn['nmt_attn']=_attn[0]
                    nmt_context=nmt_context[0][0].unsqueeze(1)[:-1]
                    setattr(my_batch, "src", (nmt_context,torch.tensor([nmt_context.size(0)])))
                    setattr(my_batch, "tgt", (batch.tgt))
                    
                    _stat, _attn, context, probs = self.translator['tgt'].translate([my_batch], model=self.model.tts)
                    stat.update(_stat)
                    attn['tts_attn']=_attn[0]
                    
                elif isinstance(self.model,OsamuraModel):
                    #import pdb; pdb.set_trace()
                    my_batch=Batch()
                    src,src_length=batch.src
                    setattr(my_batch, "src", (batch.src))
                    setattr(my_batch, "tgt", (batch.mid))
                    setattr(my_batch, 'batch_size', batch.batch_size)
                    setattr(my_batch, 'indices', batch.indices)
                    stat, attn, context, probs = self.translator['src_txt'].translate([my_batch], model=self.model.pre_model)
                    
                    setattr(my_batch, "src", (torch.exp(probs[0][0]).unsqueeze(1),torch.tensor([probs[0][0].size(0)])))
                    setattr(my_batch, "tgt", (batch.tgt))
                    #import pdb; pdb.set_trace()
                    stat, attn, context, probs = self.translator['tgt'].translate([my_batch], model=self.model.post_model)
                elif isinstance(self.model,CascadeModel):
                    pass
                stats.append(stat)
                attns.append(attn)
                cnt+=1
                if cnt>10:
                    break
        self.model.train()

        return stats, attns

    def adversarial_classifier(self, batch,normalization, result, total_stats,
                               report_stats):
        
        self.optim_ad.zero_grad()
        loss, batch_stats = self.loss_ad(
            batch,
            result,
            normalization =normalization,
            shard_size = self.shard_size
            )
        if loss is not None:
            self.optim_ad.backward(loss)
        total_stats.update(batch_stats)
        report_stats.update(batch_stats)
        self.optim_ad.step()
        

    def _gradient_accumulation(self, true_batches, normalization, total_stats,
                               report_stats):
        if self.accum_count > 1:
            self.optim.zero_grad()

        for k, batch in enumerate(true_batches):
            # 1. F-prop all but generator.
            if self.accum_count == 1:
                self.optim.zero_grad()
            
            result = self.model(batch=batch)
            
            # 3. Compute loss.
            try:
                loss, batch_stats, pred = self.train_loss(
                    batch,
                    result,
                    normalization =normalization,
                    shard_size = self.shard_size
                    )
                if loss is not None:
                    self.optim.backward(loss)
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)
            except Exception:
                traceback.print_exc()
                logger.info("At step %d, we removed a batch - accum %d",
                            self.optim.training_step, k)

            # 4. Update the parameters and statistics.
            if self.accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters()
                                if p.requires_grad
                                and p.grad is not None]
                    onmt.utils.distributed.all_reduce_and_rescale_tensors(
                        grads, float(1))
                self.optim.step()
                if self.optim_ad is not None and self.loss_ad is not None:
                    result['memory']=result['memory'].detach()
                    self.adversarial_classifier(batch,normalization,result,total_stats,
                    report_stats)
            
            self.model.reset_state()
                

        if self.accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()
        #import pdb;pdb.set_trace()
        try:
            result['pred']=torch.tensor(pred)[:-1,0].detach().cpu().numpy()#batch.tgt[0].detach().cpu().numpy()
            result['tgt']=batch.tgt[0][1:-1,0].detach().cpu().numpy()
            result['src']=batch.src[0][:,0].detach().cpu().numpy()
        except:
            pass

        return result

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return onmt.utils.Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats, attns):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats, attns,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None, attns=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats, attns=attns,)

    def _maybe_report_test(self, step, report_stats, attns):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_test(step, report_stats, attns)
