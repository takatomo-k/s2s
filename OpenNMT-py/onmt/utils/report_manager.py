""" Report manager utility """
from __future__ import print_function
import time
from datetime import datetime
import os
import onmt
import torch
import torch.nn.functional as F
from onmt.utils.logging import logger, make_log_path
from onmt.utils.evaluation import bleu, wer
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font_path = '/home/is/takatomo-k/Desktop/IPAexfont00401/ipaexg.ttf'
font_prop = FontProperties(fname=font_path,size=10)
from scipy.io import wavfile
plt.switch_backend('agg')
plt.rcParams["font.size"] = 10
from tensorboardX import SummaryWriter
from onmt.utils.audio import spectrogram2wav, n_fft, sample_rate, hop_length
import librosa.display
from onmt.inputters.text_dataset import TextTransform

def build_report_manager(opt, fields):
    tensorboard_log_dir = make_log_path(opt).replace("model/","log/")
    #import pdb; pdb.set_trace()
    print("Log directory is ",tensorboard_log_dir)
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")
    
    report_mgr = ReportMgr(fields, tensorboard_log_dir, opt.report_every, start_time=-1,
                           tensorboard_writer=writer)
    return report_mgr

class ReportMgrBase(object):
    """
    Report Manager Base class
    Inherited classes should override:
        * `_report_training`
        * `_report_step`
    """

    def __init__(self, report_every, start_time=-1.):
        """
        Args:
            report_every(int): Report status every this many sentences
            start_time(float): manually set report start time. Negative values
                means that you will need to set it later or use `start()`
        """
        self.report_every = report_every
        self.progress_step = 0
        self.start_time = start_time

    def start(self):
        self.start_time = time.time()

    def log(self, *args, **kwargs):
        logger.info(*args, **kwargs)

    def report_training(self, step, num_steps, learning_rate,
                        report_stats, attns, multigpu=False):
        """
        This is the user-defined batch-level traing progress
        report function.

        Args:
            step(int): current step count.
            num_steps(int): total number of batches.
            learning_rate(float): current learning rate.
            report_stats(Statistics): old Statistics instance.
        Returns:
            report_stats(Statistics): updated Statistics instance.
        """
        if self.start_time < 0:
            raise ValueError("""ReportMgr needs to be started
                                (set 'start_time' or use 'start()'""")
        if step % self.report_every == 0:
            if multigpu:
                report_stats = \
                    onmt.utils.Statistics.all_gather_stats(report_stats)
            self._report_training(
                step, num_steps, learning_rate, report_stats, attns)
            self.progress_step += 1
            return onmt.utils.Statistics()
        else:
            return report_stats
    
    def report_test(self, step, report_stats, attns):
       return self._report_test(step, report_stats, attns)

    def _report_training(self, *args, **kwargs):
        """ To be overridden """
        raise NotImplementedError()
    
    def _report_test(self, *args, **kwargs):
        """ To be overridden """
        raise NotImplementedError()
    
    def report_step(self, lr, step, train_stats=None, valid_stats=None, attns=None):
        """
        Report stats of a step

        Args:
            train_stats(Statistics): training stats
            valid_stats(Statistics): validation stats
            lr(float): current learning rate
        """
        self._report_step(
            lr, step, train_stats=train_stats, valid_stats=valid_stats, attns=attns)

    def _report_step(self, *args, **kwargs):
        raise NotImplementedError()
    

class ReportMgr(ReportMgrBase):
    def __init__(self, fields, path, report_every, start_time=-1., tensorboard_writer=None):
        """
        A report manager that writes statistics on standard output as well as
        (optionally) TensorBoard

        Args:
            report_every(int): Report status every this many sentences
            tensorboard_writer(:obj:`tensorboard.SummaryWriter`):
                The TensorBoard Summary writer to use or None
        """
        super(ReportMgr, self).__init__(report_every, start_time)
        self.tensorboard_writer = tensorboard_writer
        self.path = path
        self.fields=fields
    
    def maybe_log_tensorboard(self, stats, attns, prefix, learning_rate, step):
        if self.tensorboard_writer is not None:
            stats.log_tensorboard(
                prefix, self.tensorboard_writer, learning_rate, step)
            #import pdb; pdb.set_trace()
            if attns is not None:
                try:
                    if attns['ref'].shape[1] == 1025:
                            wav =self.fields['tgt'].reverse(attns['ref'])
                            self.plot_wav(wav, prefix+"/ref", step)
                            self.plot_spc(attns['ref'], prefix+"/ref", step)
                except:
                    pass
                try:
                    if attns['hyp'].shape[1] == 1025:
                            wav =self.fields['tgt'].reverse(attns['hyp'])
                            self.plot_wav(wav, prefix+"/hyp", step)
                            self.plot_spc(attns['hyp'], prefix+"/hyp", step)
                except:
                    pass
                for k in attns.keys():
                    if 'attns' in k:
                        try:
                            title=''.join(self.fields['tgt'].reverse(attns['ref']))+"@"+''.join(self.fields['tgt'].reverse(attns['hyp']))
                        except:
                            title=None
                        try:
                            self.plot_attention(attns[k]['std'][:,0], prefix+"/"+k, step,src=attns['src'],tgt=attns['hyp'],title=title)
                        except:
                            pass
    def _report_training(self, step, num_steps, learning_rate,
                         report_stats, attns):
        """
        See base class method `ReportMgrBase.report_training`.
        """
        report_stats.output(step, num_steps,
                            learning_rate, self.start_time)
        # Log the progress using the number of batches on the x-axis.
        self.maybe_log_tensorboard(report_stats,
                                   attns,
                                   "progress",
                                   learning_rate,
                                   self.progress_step)
        report_stats = onmt.utils.Statistics()

        return report_stats

    def _report_test(self, step, report_stats, attns):
        """
        See base class method `ReportMgrBase.report_training`.
        """
        #import pdb; pdb.set_trace()
        
        score= self._report_text(step, report_stats, attns) + self._report_speech(step, report_stats, attns)
        return score

    def _report_speech(self, step, report_stats, attns):
        #import pdb; pdb.set_trace()
        mel_loss=0

        for idx, (r,a) in enumerate(zip(report_stats, attns)):
            if 'pred_mel' in r and 'ref_mel' in r:
                length=min(r['ref_mel'][0].shape[0],r['pred_mel'][0].shape[0])
                mel_loss += F.smooth_l1_loss(r['pred_mel'][0][:length,0].cpu(),r['ref_mel'][0][:length,0].cpu()).item()
            if 'hyp' in a and idx<10:
                try:
                    wav=self.fields['tgt'].reverse(attns[k])
                    #import pdb; pdb.set_trace()
                    wavfile.write('/home/is/takatomo-k/'+str(idx)+'.wav',16000,wav)
                    self.tensorboard_writer.add_audio("tgtwav/"+str(1),wav,sample_rate=16000,global_step=step)
                except:
                    pass
            if 'ref' in a and idx<10:
                try:
                    wav=self.fields['ref'].reverse(attns[k])
                    #import pdb; pdb.set_trace()
                    wavfile.write('/home/is/takatomo-k/'+str(idx)+'ref.wav',16000,wav)
                    self.tensorboard_writer.add_audio("refwav/"+str(1),wav,sample_rate=16000,global_step=step)
                except:
                    pass
        return 1-mel_loss

    def _chech_type(self,report_stats):
        for r in report_stats:
            if "_" in r['hyp']:
                return "char"
        else:
            return "word"

    
    
    def _report_text(self, step, report_stats, attns):
        bleu_score=0
        word_error_rate=0
        _type="char"#self._chech_type(report_stats)
        if not os.path.exists(self.path+"/"+str(step)):
            os.makedirs(self.path+"/"+str(step))
        keys=[['asr_tgt','asr_pred'],['nmt_tgt','nmt_pred'],['tgt','pred']]
        for ref,hyp in keys:
            if ref in report_stats[0][0] and hyp in report_stats[0][0]:
                tag=ref.replace('_tgt','').replace('pred','result')
                #import pdb; pdb.set_trace()
        
                try:
                    with open(self.path+"/"+str(step)+"/"+tag+'.txt',"w") as f:
                        for idx, r in enumerate(report_stats):
                            r=r[0]
                            _bleu_score = bleu(r[ref], r[hyp], _type)
                            _word_error_rate = wer(r[ref], r[hyp], _type)
                            f.write(tag+" SENT "+str(idx)+"\t wer:"+str(_word_error_rate)+"\t bleu:"+str(_bleu_score)+"\n")
                            f.write(tag+"tgt:"+r[ref]+"\n")
                            f.write(tag+"pred:"+r[hyp]+"\n\n")
                            bleu_score+=_bleu_score
                            word_error_rate+=_word_error_rate
                        self.tensorboard_writer.add_scalar("test/"+tag+"/bleu", bleu_score/len(report_stats), step)
                        self.tensorboard_writer.add_scalar("test/"+tag+"/wer", word_error_rate/len(report_stats), step)
                        print(tag+" Test BLEU:"+str( bleu_score/len(report_stats)))
                        print(tag+" Test WER: "+str( word_error_rate/len(report_stats)))
                        bleu_score,word_error_rate=0,0
                except:
                    pass
        for i in range(len(attns)):
            for k,v in attns[i].items() :
                if k not in {"tgt","ref"}:
                    self.plot_attention(v[0], 'test/'+k, step,)
            if i>=5:
                break
        return bleu_score/len(report_stats)

    def _report_step(self, lr, step, train_stats=None, valid_stats=None,attns=None):
        """
        See base class method `ReportMgrBase.report_step`.
        """
        
        if valid_stats is not None:
            self.maybe_log_tensorboard(valid_stats,
                                       attns,
                                       "valid",
                                       lr,
                                       step)

    def plot_attention(self, score, prefix, step, title = None, src=None, tgt=None):
        #import pdb; pdb.set_trace()
        if src is not None and isinstance(self.fields['src'],TextTransform):
            src= self.fields['src'].reverse(src)
        else:
            src=None

        if tgt is not None and isinstance(self.fields['tgt'],TextTransform):
            #tgt= self.fields.reverse(attns['hyp'][:,0].detach.cpu().numpy())
            tgt= self.fields['tgt'].reverse(tgt)
        else:
            tgt=None

        if isinstance(score,torch.Tensor):
            score=score.detach().cpu().numpy()#[:,1:-1]
        fig, ax = plt.subplots(figsize=(20, 10))
        im = ax.imshow(
            score,
            aspect='auto',
            origin='lower',
            interpolation='none')
        
        if tgt is not None:
            y=tgt
            ax.set_yticks(range(len(y)))
            ax.set_yticklabels(y,fontproperties=font_prop,rotation=90)

        if src is not None:
            x = src
            ax.set_xticks(range(len(x)))
            ax.set_xticklabels(x, fontproperties=font_prop)
        
        if title is not None:
            plt.title(title, fontproperties=font_prop)
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.gca().invert_yaxis()
        self.tensorboard_writer.add_figure(prefix+"/attns", fig, step)

    def plot_wav(self, wav, prefix, step, title = None, src=None, tgt=None):
        fig = plt.figure()
        spectrum, freqs, t, im = plt.specgram(wav.T,
                                      NFFT=n_fft,
                                      Fs=sample_rate,
                                      noverlap=n_fft-hop_length,
                                      mode='magnitude')

        plt.xlabel('Time[sec]')
        plt.ylabel('Frequency[Hz]')
        fig.colorbar(im).set_label('[dB]')
        plt.tight_layout()
        self.tensorboard_writer.add_figure(prefix+"/wav", fig, step)
        self.tensorboard_writer.add_audio(prefix,wav,sample_rate=16000,global_step=step)

    def plot_spc(self, specgram, prefix, step, title = None, src=None, tgt=None):
        fig = plt.figure()
        librosa.display.specshow(specgram.T, sr=sample_rate,fmax=8000)
        plt.xlabel('Time[sec]')
        plt.ylabel('Frequency[Hz]')
        plt.tight_layout()
        self.tensorboard_writer.add_figure(prefix+"/specgram", fig, step)
            
        
