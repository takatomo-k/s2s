""" Onmt NMT Model base class definition """
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from onmt.utils import GaussianNoise
import copy

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
    
    def encode(self, src, src_lengths=None):
        return src
    
    def decode(self, tgt, memory=None, memory_lengths=None, tgt_lengths=None,step=None):
        return tgt
    
    def update_dropout(self, dropout):
        pass
    
    def reset_state(self):
        pass

class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        #self.decoder.noise.sigma=0.2

    def forward(self, src=None, tgt=None, src_lengths=None, tgt_lengths=None, bptt=False,batch=None):
        if batch is not None:
            src, src_lengths = batch.src
            tgt, tgt_lengths = batch.tgt
            if hasattr(batch,'spkr'):
                tgt[0]=self.spkr_emb(batch.spkr[0])
        
        enc_state, memory_bank, memory_lengths = self.encode(src, src_lengths)
        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(tgt[:-1], memory_bank,memory_lengths=memory_lengths, tgt_lengths =tgt_lengths)
        
        return {"dec_out": dec_out, "attns": attns,"memory":memory_bank,"lengths":memory_lengths}
    
    def encode(self, src, src_lengths):
        return self.encoder(src, src_lengths)
    
    def decode(self, tgt, memory, memory_lengths, tgt_lengths,step):
        return self.decoder(tgt, memory,
                                      memory_lengths=memory_lengths, step=step, tgt_lengths =tgt_lengths)
    
    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
    
    def reset_state(self):
        if self.decoder.state is not None:
            self.decoder.detach_state()

class ASRModel(NMTModel):
    def forward(self, mfc=None, src_txt=None, mfc_lengths=None, src_txt_lengths=None, spkr=None, bptt=False):
        enc_state, memory_bank, memory_lengths = self.encode(mfc, src_txt_lengths)
        if bptt is False:
            self.decoder.init_state(mfc, memory_bank, enc_state)
        dec_out, attns = self.decoder(src_txt[:-1], memory_bank,
                                      memory_lengths=memory_lengths, tgt_lengths =src_txt_lengths)
        return {"asr_dec_out": dec_out, "asr_attns": attns,"asr_memory":memory_bank,"asr_mem_lengths":memory_lengths}
    
class TTSModel(NMTModel):
    def forward(self, txt=None, wld=None, txt_lengths=None, wld_lengths=None, spkr=None, bptt=False):
        enc_state, memory_bank, memory_lengths = self.encode(txt, txt_lengths)
        if bptt is False:
            self.decoder.init_state(txt, memory_bank, enc_state)
        dec_out, attns = self.decoder(wld[:-1], memory_bank,
                                      memory_lengths=memory_lengths, tgt_lengths =wld_lengths)

        return {"tts_dec_out": dec_out, "tts_attns": attns,"tts_memory":memory_bank,"tts_mem_lengths":memory_lengths}


class CascadeModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, models):
        super(CascadeModel, self).__init__()
        
        self.asr = models[0]
        self.nmt = models[1]
        self.tts = models[2]
        
    def forward(self, src=None, mid=None, tgt=None, src_lengths=None, mid_lengths=None, tgt_lengths=None, bptt=False, batch=None):
        
        if batch is not None:
            src,src_lengths=batch.src
            src_txt,src_txt_lengths=batch.src_txt
            tgt_txt,tgt_txt_lengths=batch.tgt_txt
            tgt, tgt_lengths = batch.tgt

        asr_result = self.asr(src, src_txt, src_lengths, src_txt_lengths)
        asr_txt = self.asr.generator(asr_result["dec_out"])

        if isinstance(asr_txt, tuple):
            asr_txt = asr_txt[0]
        asr_txt = asr_txt.argmax(-1).unsqueeze(-1)
        #import pdb; pdb.set_trace()
        nmt_result = self.nmt_model(asr_txt[:-1], tgt_txt, src_txt_lengths, tgt_txt_lengths)
        nmt_txt = self.nmt.generator(nmt_result["dec_out"])
        if isinstance(nmt_txt, tuple):
            nmt_txt = nmt_txt[0]
        nmt_txt = nmt_txt.argmax(-1).unsqueeze(-1)
        
        tts_result=self.tts(nmt_txt, tgt, tgt_txt_lengths,tgt_lengths)
        return tts_result

    def update_dropout(self, dropout):
        self.asr.update_dropout(dropout)
        self.nmt.update_dropout(dropout)
        self.tts.update_dropout(dropout)

    def reset_state(self):
        self.asr.reset_state()
        self.nmt.reset_state()
        self.tts.reset_state()

class TranscoderModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """
    def __init__(self, models):
        super(TranscoderModel, self).__init__()
        self.asr = models[0]
        self.nmt= models[1]
        self.tts = models[2]
        self.tts_encoder = copy.deepcopy(models[2].encoder)
        self.nmt_encoder = copy.deepcopy(models[1].encoder)
        self.nmt.encoder.embeddings=None #transcoder
        self.tts.encoder.embeddings=None #transcoder
        
        self.cuda()

    def forward(self, src=None, tgt=None, src_lengths=None, tgt_lengths=None, bptt=False, batch=None):
        #import pdb; pdb.set_trace()
        self.asr.decoder.noise.sigma=0.3
        self.nmt_encoder.eval()
        self.tts_encoder.eval()
        if batch is not None:
            src,src_lengths=batch.src
            src_txt, src_txt_lengths=batch.src_txt
            tgt_txt, tgt_txt_lengths = batch.tgt_txt
            tgt, tgt_lengths = batch.tgt
        
        #import pdb; pdb.set_trace()
        result={}
        with torch.no_grad():
            self.asr.eval()
            asr_results=self.asr(src,src_txt,src_lengths,src_txt_lengths)
            result['asr_dec_out']=asr_results['dec_out']
            result['asr_attns']=asr_results['attns']

        _, memory_bank, memory_lengths = self.nmt.encode(asr_results['attns']['context'][:-1].detach(), src_txt_lengths-2)
        #nmt_results = self.nmt(asr_results['attns']['context'][:-1].detach(), tgt_txt, src_txt_lengths, tgt_txt_lengths)
        result['trans']=memory_bank
        with torch.no_grad():
            _, nmt_memory, _ = self.nmt_encoder(src_txt[1:-1], src_txt_lengths-2)
            result['trans_tgt'] = nmt_memory.detach()
            if F.smooth_l1_loss(result['trans'],result['trans_tgt']).item() > 0.005:
                memory_bank=nmt_memory
                #return result

        dec_out, attns = self.nmt.decoder(tgt_txt[:-1], memory_bank,memory_lengths=memory_lengths, tgt_lengths =tgt_txt_lengths)
        result['nmt_dec_out']=dec_out
        result['nmt_attns']=attns
        
        _, memory_bank, memory_lengths = self.tts.encoder(result['nmt_attns']['context'][:-1], tgt_txt_lengths)
        result['trans2']=memory_bank
        with torch.no_grad():
            _, tts_memory, _ = self.tts_encoder(tgt_txt[1:-1],tgt_txt_lengths-2)
            result['trans2_tgt'] = tts_memory.detach()
            if F.smooth_l1_loss(result['trans2'],result['trans2_tgt']).item() > 0.005:
                memory_bank=nmt_memory
                #return result
        dec_out,attns = self.tts.decoder(tgt[:-1], memory_bank,memory_lengths=memory_lengths, tgt_lengths =tgt_lengths)
        result['tts_dec_out']=dec_out
        result['tts_attns']=attns
        return result

    def update_dropout(self, dropout):
        self.asr.update_dropout(dropout)
        self.nmt.update_dropout(dropout)
        self.tts.update_dropout(dropout)
        
    def reset_state(self):
        self.asr.reset_state()
        self.nmt.reset_state()
        self.tts.reset_state()

class TwopassModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """
    def __init__(self, models):
        super(TwopassModel, self).__init__()
        self.asr = models[0]
        self.nmt= models[1]
        self.tts = models[2]
        self.nmt.encoder.embeddings=None
        self.tts.encoder.embeddings=None
        """
        self.tts_encoder = copy.deepcopy(models[2].encoder)
        self.nmt_encoder = copy.deepcopy(models[1].encoder)
        
        """
        self.cuda()

    def forward(self, src=None, tgt=None, src_lengths=None, tgt_lengths=None, bptt=False, batch=None):
        #import pdb; pdb.set_trace()
        if batch is not None:
            src,src_lengths=batch.src
            src_txt, src_txt_lengths=batch.src_txt
            tgt_txt, tgt_txt_lengths = batch.tgt_txt
            tgt, tgt_lengths = batch.tgt
            #spkr = self.spkr_emb(tgt[0])
            
        #print(src.shape, tgt.shape, max(src_lengths) ,max(tgt_lengths))
        #import pdb; pdb.set_trace()
        result={}
        asr_results=self.asr(src,src_txt,src_lengths,src_txt_lengths)
        result['asr_dec_out']=asr_results['dec_out']
        result['asr_attns']=asr_results['attns']

        nmt_results = self.nmt(asr_results['attns']['context'][:-1], tgt_txt, src_txt_lengths, tgt_txt_lengths)
        result['nmt_dec_out']=nmt_results['dec_out']
        result['nmt_attns']=nmt_results['attns']
        

        tts_results = self.tts(nmt_results['attns']['context'][:-1], tgt, tgt_txt_lengths, tgt_lengths)
        
        result['tts_dec_out']=tts_results['dec_out']
        result['tts_attns']=tts_results['attns']
        
        """
        result['trans']=nmt_results['memory']
        result['trans2']=tts_results['memory']
        with torch.no_grad():
            self.nmt.encoder.eval()
            self.tts.encoder.eval()
            _, nmt_memory, _ = self.nmt_encoder(src_txt[1:-1],src_txt_lengths)
            _, tts_memory, _ = self.tts_encoder(tgt_txt[1:-1],tgt_txt_lengths)
            result['trans_tgt'] = nmt_memory.detach()
            result['trans2_tgt'] = tts_memory.detach()
        #import pdb; pdb.set_trace()
        """
        return result

    def update_dropout(self, dropout):
        self.asr.update_dropout(dropout)
        self.nmt.update_dropout(dropout)
        #self.tts.update_dropout(dropout)
        
    def reset_state(self):
        self.asr.reset_state()
        self.nmt.reset_state()
        self.tts.reset_state()

class AttentionPassingModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """
    def __init__(self, models):
        super(AttentionPassingModel, self).__init__()
        self.asr = models[0]
        #self.joint=nn.Linear(256,256)
        self.nmt= models[1]
        self.nmt.encoder.embeddings=None
        self.cuda()

    def forward(self, src=None, tgt=None, src_lengths=None, tgt_lengths=None, bptt=False, batch=None):
        #import pdb; pdb.set_trace()
        if batch is not None:
            src,src_lengths=batch.src
            src_txt, src_txt_lengths=batch.src_txt
            tgt, tgt_lengths = batch.tgt
        #import pdb; pdb.set_trace()
        result={}
        enc_state,memory,memory_lengths = self.asr.encoder(src, src_lengths)
        self.asr.decoder.init_state(src, memory, enc_state)
        dec_out, attns = self.asr.decoder(src_txt[:-1], memory,
                                      memory_lengths=memory_lengths, tgt_lengths =src_txt_lengths)
        result['asr_dec_out']=dec_out
        result['asr_attns']=attns
        enc_state, memory, memory_lengths = self.nmt.encoder(attns['context'][:-1], src_txt_lengths)
        dec_out, attns = self.nmt.decoder(tgt[:-1], memory,
                                      memory_lengths=memory_lengths, tgt_lengths =tgt_lengths)
        result['nmt_dec_out']=dec_out
        result['nmt_attns']=attns
        return result

    def update_dropout(self, dropout):
        self.asr.update_dropout(dropout)
        #self.nmt.update_dropout(dropout)
        #self.tts.update_dropout(dropout)
        
    def reset_state(self):
        self.asr.reset_state()
        self.nmt.reset_state()
        #self.tts.reset_state()
        
class CunstomLinear(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """
    def __init__(self, insize, outsize):
        super(CunstomLinear, self).__init__()
        self.linear=nn.Linear(insize,outsize)
        self.dropout=nn.Dropout(p=0.1)
        self.cuda()
    
    def forward(self,x):
        return self.linear(x)

    def update_dropout(self, dropout):
        self.dropout=nn.Dropout(dropout)
        


class SpeechAECModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """
    def __init__(self, models):
        super(SpeechAECModel, self).__init__()
        self.encoder = models[0]
        self.decoder = models[1].decoder
        
        self.decoder.noise.sigma = 0.2
        self.spkr_emb = nn.Embedding(2,400)
        self.joint_layer= nn.Linear(256,512)
        self.cuda()

    def forward(self, src=None, tgt=None, src_lengths=None, tgt_lengths=None, bptt=False, batch=None):
        #import pdb; pdb.set_trace()
        if batch is not None:
            src,src_lengths=batch.src
            txt, txt_lengths=batch.txt
            spkr,_ = batch.spkr
            tgt,tgt_lengths=batch.tgt
            tgt[0]=self.spkr_emb(spkr)
        #print(src.shape, tgt.shape, max(src_lengths) ,max(tgt_lengths))
        #import pdb; pdb.set_trace()
    
        result = self.encoder(src, txt, src_lengths ,txt_lengths)
        #result['attns']['std']=self.joint_layer(result['attns']['context'])
        #return result
        #dec_out, attns = self.decoder(tgt[:-1], result['attns']['context'][:-1], memory_lengths=txt_lengths, tgt_lengths =tgt_lengths)

        dec_out, attns = self.decoder(tgt[:-1], self.joint_layer(result['attns']['context'][:-1]), memory_lengths=txt_lengths, tgt_lengths =tgt_lengths)
        
        result['txt_out'] = result['dec_out']
        result['tts_attns'] = attns
        result['dec_out']= dec_out
        return result

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        if self.decoder.state is not None:
            self.decoder.detach_state()
    def reset_state(self):
        self.encoder.reset_state() 

class GoogleMultitaskModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """
    def __init__(self, models):
        super(GoogleMultitaskModel, self).__init__()
        self.encoder = models[0]
        self.nmt_decoder = models[1].decoder
        setattr(self.nmt_decoder,'generator',models[1].generator)
        self.decoder = models[2].decoder
        #self.decoder.noise.sigma=0.2
        #self.spkr_emb = nn.Embedding(2,512)
        self.cuda()

    def forward(self, src=None, tgt=None, src_lengths=None, tgt_lengths=None, bptt=False, batch=None):
        #import pdb; pdb.set_trace()
        if batch is not None:
            src,src_lengths=batch.src
            tgt, tgt_lengths = batch.tgt

            src_txt, src_txt_lengths=batch.src_txt
            tgt_txt, tgt_txt_lengths = batch.tgt_txt
 
            #spkr = self.spkr_emb(tgt[0])
            
        #print(src.shape, tgt.shape, max(src_lengths) ,max(tgt_lengths))
        #import pdb; pdb.set_trace()
        result = self.encoder(src, src_txt, src_lengths ,src_txt_lengths)
        result['asr_attns']=result['attns']
        result['src_txt_out'] = result['dec_out']
        dec_out, attns = self.nmt_decoder(tgt_txt[:-1], result['memory'], memory_lengths=result['lengths'], tgt_lengths =tgt_txt_lengths)
        result['nmt_dec_out'] = dec_out
        result['nmt_attns']=attns
        #result['memory']=torch.cat((spkr,result['memory']))
        dec_out, attns = self.decoder(tgt[:-1], result['memory'], memory_lengths=result['lengths']+1, tgt_lengths =tgt_lengths)
        result['tts_attns'] = attns
        result['dec_out']= dec_out
        return result

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        if self.nmt_decoder.state is not None:
            self.nmt_decoder.detach_state()
        if self.decoder.state is not None:
            self.decoder.detach_state()
    
    def reset_state(self):
        self.encoder.reset_state() 


class NeuralCascadeModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """
    def __init__(self, models):
        super(NeuralCascadeModel, self).__init__()
        self.asr = models[0]
        self.asr_generator=models[0].generator
        self.nmt = models[1]
        self.cuda()

    def forward(self, src=None, tgt=None, src_lengths=None, tgt_lengths=None, bptt=False, batch=None):
        #import pdb; pdb.set_trace()
        result={}
        if batch is not None:
            src,src_lengths=batch.src
            src_txt, src_txt_lengths=batch.src_txt
            tgt_txt, tgt_txt_lengths = batch.tgt_txt
            
        asr_result = self.asr(src, src_txt, src_lengths ,src_txt_lengths)
        result['asr_attns']=asr_result['attns']
        result['asr_dec_out'] = result['dec_out']
        asr_txt=torch.softmax(self.asr_generator(result['dec_out']))
        nmt_result = self.nmt(tgt_txt[:-1], result['dec_out'], memory_lengths=src_txt_lengths, tgt_lengths =tgt_txt_lengths)
        result['nmt_dec_out'] = nmt_result['dec_out']
        result['nmt_attns']=nmt_result['attns']
        return result

    def update_dropout(self, dropout):
        self.asr.update_dropout(dropout)
        self.nmt.update_dropout(dropout)

    def reset_state(self):
        self.asr.reset_state() 
        self.asr.reset_state() 


class ModelPostNet(nn.Module):
    """
    CBHG Network (mel --> linear)
    """
    def __init__(self, encoder, decoder):
        super(ModelPostNet, self).__init__()
        self.encoder = None
        self.decoder = decoder
        

    def forward(self, mel=None, bptt=False,batch=None):
        if batch is not None:
            mel,_=batch.src
            l,b,d=mel.size()
            mel=mel.view(-1,b,80)
        

        mag = self.decoder(mel.transpose(1,2))
        
        return {"dec_out": mag, "attns": None,"memory":None,"lengths":None}
    
    def update_dropout(self, dropout):
        pass

    def reset_state(self):
        pass