"""
This includes: LossComputeBase and the standard TextLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import onmt
import onmt.inputters
from onmt.modules.sparse_losses import SparsemaxLoss
from onmt.modules.sparse_activations import LogSparsemax
from onmt.inputters.audio_dataset import AudioTransform
from onmt.inputters.text_dataset import TextTransform
from onmt.utils.misc import sequence_mask
from onmt.utils.gaussian_noise import GaussianNoise



def build_loss_compute(model, opt, train=True):
    """
    Returns a LossCompute subclass which wraps around an nn.Module subclass
    (such as nn.NLLLoss) which defines the loss criterion. The LossCompute
    object allows this loss to be computed in shards and passes the relevant
    data to a Statistics object which handles training/validation logging.
    Currently, the TextLossCompute class handles all loss computation except
    for when using a copy mechanism.
    """
    device = torch.device("cuda" if onmt.utils.misc.use_gpu(opt) else "cpu")
    #import pdb;pdb.set_trace()
    compute = str2loss[opt.model_type](model)
    compute.to(device)
    return compute


def filter_shard_state(state, shard_size=None):
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.
    Yields:
        Each yielded shard is a dict.
    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        
        non_none = dict(filter_shard_state(state, shard_size))

        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)

class LossComputeBase(nn.Module):
    def __init__(self):
        super(LossComputeBase, self).__init__()

    @property
    def padding_idx(self):
        return self.criterion.ignore_index

    def _make_shard_state(self, batch, range_, **kwargs):
        return NotImplementedError

    def _compute_loss(self, output, target, **kwargs):
        return NotImplementedError

    def __call__(self,
                 batch,
                 result,
                 normalization=1.0,
                 shard_size=0,
                 ):
        trunc_range = (0, batch.tgt[0].size(0))
        shard_state = self._make_shard_state(batch, trunc_range, result)
        if shard_size == 0:
            loss, stats, hyp  = self._compute_loss(**shard_state)
            stats=onmt.utils.Statistics(stats) 
            return loss / float(normalization), stats, hyp
        batch_stats = onmt.utils.Statistics()
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(**shard)
            stats=onmt.utils.Statistics(batch_stats) 
            loss.div(float(normalization)).backward()
            batch_stats.update(stats)
        return None, stats

    def _stats(self, loss, scores, target, stats={}, tag=""):
        #import pdb;pdb.set_trace()
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()
        stats[tag+'xent']=loss
        stats[tag+'acc']=100*(num_correct/num_non_padding)
        return stats

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')

class StopLossCompute(LossComputeBase):
    def __init__(self):
        self.criterion=nn.BCELoss()

    def _make_shard_state(self, batch, range_, result):
        #import pdb;pdb.set_trace()
        tgt,tgt_lengths = batch.tgt
        return {
            "output": result["dec_out"],
            "target": tgt[1:],
        }


class TextLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """
    def __init__(self, model, normalization="sents"):
        super(TextLossCompute, self).__init__()
        self.criterion = nn.NLLLoss(ignore_index=0)
        self.generator = model.generator 
        
    def _make_shard_state(self, batch, range_, result):
        tgt,tgt_lengths = batch.tgt
        return {
            "output": result["dec_out"],
            "target": tgt[1:],
        }
    
    def _compute_loss(self, output, target, tag=""):
        l,b,d=target.size()
        gtruth = target.view(-1)
        bottled_output = self._bottle(output)
        scores = self.generator(bottled_output) if self.generator is not None else bottled_output
        
        if isinstance(scores, tuple):
            loss_1st = self.criterion(scores[0], gtruth)
            loss_2nd = self.criterion(scores[1], gtruth)
            loss= loss_1st+loss_2nd
            stats = self._stats(loss_1st.item() ,scores[0], gtruth, tag=tag)
            stats = self._stats(loss_2nd.item() ,scores[1], gtruth, stats=stats, tag=tag+'acoustic_')
            scores=scores[0]
        else:
            loss = self.criterion(scores, gtruth)
            stats = self._stats(loss.item(), scores, gtruth, tag=tag)
        #import pdb; pdb.set_trace()
        return loss, stats, scores.view(l,b,-1).argmax(-1)
    
    def _stats(self, loss, scores, target, stats={},tag=""):
        if isinstance(scores, tuple):
            stats=super()._stats(loss, scores[0], target, stats,tag=tag)
            stats=super()._stats(loss, scores[1], target, stats,tag="2nd_"+tag)
        else:
            stats=super()._stats(loss, scores, target, stats,tag=tag)
        return stats

class SpeechLossCompute(LossComputeBase):

    def __init__(self, model, normalization="sents"):
        super(SpeechLossCompute, self).__init__()
        self.criterion = nn.MSELoss()
        self.stop_loss=nn.BCELoss()
        self.generator = model.generator
        
    def _make_shard_state(self, batch, range_, result):
        mel, mel_lengths = batch.tgt
        #import pdb;pdb.set_trace()
        return {
            "output": result["dec_out"],
            "target": mel[1:],
            "lengths": sequence_mask(mel_lengths-1).transpose(0,1).unsqueeze(-1).type(torch.FloatTensor).cuda()
        }

    def _compute_loss(self, output, target, lengths, tag=''):
        #import pdb;pdb.set_trace()
        l, b, d= output.size()
        mel, post_mel, stops = self.generator(output)
        mel_loss = self.criterion(mel.view(-1, b, 80), target.view(-1, b, 80))
        post_mel_loss = self.criterion(post_mel, target.view(-1, b, 80))
        stop_loss= self.stop_loss(stops, lengths)
        stats = self._stats(mel_loss.item(), post_mel_loss.item(), stop_loss.item())
        return mel_loss+post_mel_loss+stop_loss, stats, mel

    def _stats(self, mel_loss, post_mel_loss, stop_loss, std_loss=None,mean_loss=None, stats={},tag=''):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        stats['mel_loss']=mel_loss
        stats['post_mel_loss']=post_mel_loss
        stats['stop_loss']=stop_loss
        return stats

class AdversarialLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, generator,criterion):
        super(AdversarialLossCompute, self).__init__()
        self.criterion = criterion
        self.generator = generator
        

    def _make_shard_state(self, batch, range_, result):
        spkr,_ = batch.spkr
        #import pdb;pdb.set_trace()
        return {
            "output": result["memory"],
            "lengths": result["lengths"],
            "speaker": spkr#.expand(result["memory"].size(0), result["memory"].size(1))
        }

    def _compute_loss(self, output, speaker,lengths):
        
        #bottled_output = self._bottle(output)
        scores = self.generator(output,lengths)
        gtruth = speaker.contiguous().view(-1)
        loss = self.criterion(scores, gtruth)
        stats = self._stats(loss.item(), scores)
        return loss, stats

    def _stats(self, loss, scores,stats={}):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets
        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        #import pdb;pdb.set_trace()
        stats['spkr_loss']=loss

        return onmt.utils.Statistics(stats)

class CascadeLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, generators, normalization="sents"):
        super(CascadeLossCompute, self).__init__()
        self.asr_loss = TextLossCompute(model.encoder)
        self.nmt_loss = TextLossCompute(model.decoder)
    
    def _make_shard_state(self, batch, range_, result):
        #import pdb;pdb.set_trace()
        src_txt = batch.src_txt[0]
        tgt = batch.tgt[0]
        return {
            "output": result["dec_out"],
            "target": tgt[1:],
            "src_txt_out": result["src_txt_out"],
            "src_txt": src_txt[1:]
        }

    def _compute_loss(self, output, target, src_txt_out, src_txt,tag=''):
        asr_loss, asr_stats=self.asr_loss._compute_loss(src_txt_out,src_txt,'asr')
        nmt_loss, nmt_stats=self.nmt_loss._compute_loss(output, target,'nmt')

        return asr_loss+nmt_loss, nmt_stats.update(asr_stats)


class TranscodeLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, model, normalization="sents",tag=""):
        super(TranscodeLossCompute, self).__init__()
        self.trans_loss = nn.MSELoss()
        self.sp_loss  = SpeechLossCompute(model)
        self.asr_loss = TextLossCompute(model.asr)
        self.nmt_loss = TextLossCompute(model.nmt)
    
    def _make_shard_state(self, batch, range_, result):
        #import pdb;pdb.set_trace()
        src_txt=batch.src_txt[0]
        ret={"asr_out": result["asr_dec_out"],
            "src_txt":src_txt[1:],"trans":result["trans"],
            "trans_tgt": result["trans_tgt"],
            }
        if "nmt_dec_out" in result: 
            tgt_txt=batch.tgt_txt[0]
            ret["nmt_out"]  =result["nmt_dec_out"]
            ret["tgt_txt"]= tgt_txt[1:]
            ret["trans2"] =result["trans2"]
            ret["trans2_tgt"]=result["trans2_tgt"]
        
        if "tts_dec_out" in result:
            tgt,tgt_lengths = batch.tgt
            ret["tts_out"]=result["tts_dec_out"]
            ret["target"]=tgt[1:]
            ret["target_lengths"]=sequence_mask(tgt_lengths).transpose(0,1).unsqueeze(-1).type(torch.FloatTensor).cuda()
        
        
        
        return ret

    def _compute_loss(self, asr_out,src_txt,trans, trans_tgt,nmt_out=None,tgt_txt=None, tts_out=None, target=None, target_lengths=None,  trans2=None, trans2_tgt=None, tag=''):
        #import pdb; pdb.set_trace()
        asr_loss, asr_stats, hyp = self.asr_loss._compute_loss(asr_out,src_txt, tag=tag+'asr')
        trans_loss = self.trans_loss(trans,trans_tgt)
        asr_stats['trans_loss']=trans_loss.item()
        
        if nmt_out is not None:
            nmt_loss, nmt_stats, hyp = self.nmt_loss._compute_loss(nmt_out,tgt_txt, tag=tag+'nmt')
            trans2_loss=self.trans_loss(trans2,trans2_tgt)
            asr_stats.update(nmt_stats)
            asr_stats['trans2_loss']=trans2_loss.item()
        else:
            return  asr_loss+trans_loss, asr_stats, hyp
        
        if tts_out is not None:
            #import pdb; pdb.set_trace()
            tts_loss, tts_stats, hyp = self.sp_loss._compute_loss(tts_out ,target, target_lengths)
            asr_stats.update(tts_stats)
        else:
            return asr_loss+trans_loss+nmt_loss+trans2_loss, asr_stats, hyp
        
        return asr_loss+trans_loss+nmt_loss+trans2_loss+tts_loss, asr_stats, hyp

    def _stats(self, trans_loss, stats={},tag=""):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        #import pdb; pdb.set_trace()
        stats['trans_loss']=trans_loss
        return stats

class APassingLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, model, normalization="sents",tag=""):
        super(APassingLossCompute, self).__init__()
        self.trans_loss = nn.SmoothL1Loss()
        self.asr_loss = TextLossCompute(model.asr)
        self.nmt_loss = TextLossCompute(model)
    
    def _make_shard_state(self, batch, range_, result):
        #import pdb;pdb.set_trace()
        src_txt=batch.src_txt[0]
        tgt,tgt_lengths = batch.tgt

        return {
            "asr_out": result["asr_dec_out"],
            "src_txt":src_txt[1:],
            "nmt_out": result["nmt_dec_out"],
            "tgt_txt": tgt[1:],
        }

    def _compute_loss(self, asr_out,src_txt,nmt_out,tgt_txt, tag=''):
        
        asr_loss, asr_stats = self.asr_loss._compute_loss(asr_out,src_txt, tag=tag+'asr')
        nmt_loss, nmt_stats = self.nmt_loss._compute_loss(nmt_out,tgt_txt, tag=tag+'nmt')
        asr_stats.update(nmt_stats)
        
        return asr_loss+nmt_loss, asr_stats 

    def _stats(self, trans_loss, stats={},tag=""):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        #import pdb; pdb.set_trace()
        stats['trans_loss']=trans_loss
        return stats

class EmphasisLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, xent, generator, normalization="sents",tag=""):
        super(EmphasisLossCompute, self).__init__()
        self.emphasis_loss = nn.NLLLoss(ignore_index=3)
        self.txt_loss = TextLossCompute(xent,generator,tag="nmt_")
        self.tag=tag
        
    def _make_shard_state(self, batch, range_, result):
        #import pdb;pdb.set_trace()
        tgt = batch.tgt[0]
        mid = batch.mid[0]
        #import pdb;pdb.set_trace()
        return {
            "output": result["dec_out"],
            "target": tgt[1:],
            "middle": mid[1:],
        }

    def _compute_loss(self, output, target, mid_out, middle, trans, trans_tgt):
        #import pdb;pdb.set_trace()
        txt_loss,txt_stats = self.txt_loss._compute_loss(batch,output,target)
        self.emphasis_loss
        stats = self._stats(nmt_stats,trans_stats,tag=self.tag)
        return trans_loss+nmt_loss, stats
    
    def _stats(self, nmt_stats, stats={},tag=""):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        stats.update(nmt_stats)
        return stats

class SpeechAECLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, model, normalization="sents"):
        super(SpeechAECLossCompute, self).__init__()
        self.sp_loss = SpeechLossCompute(model)
        self.txt_loss = TextLossCompute(model.encoder)
        
    def _make_shard_state(self, batch, range_, result):
        #import pdb;pdb.set_trace()
        tgt, tgt_lengths = batch.tgt
        txt = batch.txt[0]
        
        #import pdb;pdb.set_trace()
        return {
            "output": result["dec_out"],
            "target": tgt[1:],
            "tgt_lengths":sequence_mask(tgt_lengths).transpose(0,1).unsqueeze(-1).type(torch.FloatTensor).cuda(),
            "txt_out": result["txt_out"],
            "txt": txt[1:]
           }

    def _compute_loss(self, output, target, tgt_lengths, txt_out, txt, tag="" ):
        txt_loss,txt_stats = self.txt_loss._compute_loss(txt_out, txt)
        sp_loss,sp_stats = self.sp_loss._compute_loss(output ,target, tgt_lengths)
        sp_stats.update(txt_stats)
        return txt_loss+sp_loss, sp_stats


class GoogleMultitaksLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, model, normalization="sents"):
        super(GoogleMultitaksLossCompute, self).__init__()
        self.sp_loss  = SpeechLossCompute(model)
        self.asr_loss = TextLossCompute(model.encoder)
        self.nmr_loss = TextLossCompute(model.nmt_decoder)
        
    def _make_shard_state(self, batch, range_, result):
        tgt, tgt_lengths = batch.tgt
        src_txt = batch.src_txt[0]
        tgt_txt = batch.tgt_txt[0]

        return {
            "output": result["dec_out"],
            "target": tgt[1:],
            "tgt_lengths":sequence_mask(tgt_lengths).transpose(0,1).unsqueeze(-1).type(torch.FloatTensor).cuda(),
            "asr_dec_out": result["asr_dec_out"],
            "src_txt": src_txt[1:],
            "tgt_txt_out": result["tgt_txt_out"],
            "tgt_txt":tgt_txt[1:],
            
           }

    def _compute_loss(self, output, target, tgt_lengths, src_txt_out, src_txt, tgt_txt_out, tgt_txt, tag=''):
        #import pdb;pdb.set_trace()
        #output, target, lengths, linear_tgt=None
        asr_loss,asr_stats, hyp = self.asr_loss._compute_loss(src_txt_out, src_txt,tag='asr')
        nmt_loss,nmt_stats, hyp = self.nmr_loss._compute_loss(tgt_txt_out, tgt_txt,tag='nmt')
        sp_loss,sp_stats, hyp = self.sp_loss._compute_loss(output ,target, tgt_lengths )
        asr_stats.update(nmt_stats)
        asr_stats.update(sp_stats)
        return asr_loss+nmt_loss+sp_loss, asr_stats, hyp

class LinearLossCompute(LossComputeBase):

    def __init__(self, model, normalization="sents"):
        super(LinearLossCompute, self).__init__()
        self.criterion = nn.SmoothL1Loss()
        self.generator = model.generator

    def _make_shard_state(self, batch, range_, result):

        return {"output": result["dec_out"],"post_mel": result["memory"], "mel": batch.src[0], "target": batch.tgt[0] }

    def _compute_loss(self, output, post_mel, mel,  target, tag=''):
        #import pdb;pdb.set_trace()
        mag=self.generator(output)
        mag_loss = self.criterion( mag[:target.shape[0]], target)
        return mag_loss, self._stats(mag_loss.item()), mag

    def _stats(self, mag_loss, stats={},tag=''):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        stats['mag_loss'] =mag_loss
        return stats

str2loss={'single':TextLossCompute,'nmt':TextLossCompute,'asr':TextLossCompute,'tts':SpeechLossCompute,'cascade':CascadeLossCompute,
          'transcode':TranscodeLossCompute,'spaec':SpeechAECLossCompute, 'google':GoogleMultitaksLossCompute,'ncascade':GoogleMultitaksLossCompute,'ap':APassingLossCompute,'vocoder':LinearLossCompute}
