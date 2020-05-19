import os
import torch
import onmt
import torch.nn as nn
from onmt.modules.util_class import Cast
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from onmt.modules import PostConvNet, Conv, CBHG
from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory


    

class TextGenerator(nn.Module):
    """
    A single layer of the text output.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """
    def __init__(self, model_opt, field, decoder ,stop=True):
        super(TextGenerator, self).__init__()
        self.gen_type = model_opt.gen_type
        if not model_opt.copy_attn:
            if model_opt.generator_function == "sparsemax":
                gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
            else:
                gen_func = nn.LogSoftmax(dim=-1)
            self.out = nn.Linear(model_opt.dec_rnn_size, len(field))
            self.gen_func = nn.Sequential(                
                Cast(torch.float32),
                gen_func
            )
            if model_opt.share_dec_emb or self.gen_type !=0:
                self.out.weight = decoder.embeddings.word_lut.weight
        else:
            import pdb; pdb.set_trace()

        self.eos = field.eos_idx
        if self.gen_type in {1,2,3}:
            self.tts_embed_weight = torch.load(model_opt.add_word_vecs_dec)['model']['encoder.embeddings.embed.make_embedding.emb_luts.0.weight']
        if self.gen_type in {3,4}:
            self.bi_out = nn.Bilinear(len(field), len(field), len(field))
        if self.gen_type == 4:
            self.extra_out = nn.Linear(model_opt.dec_rnn_size, len(field))
    
    def forward(self, inputs):
        text_out = self.out(inputs)
        if self.gen_type > 0:
            if self.gen_type == 4:
                return self.gen_func(self.bi_out(text_out, self.extra_out(inputs)))
            tts_out = F.linear(inputs, self.tts_embed_weight)
            if self.gen_type == 3:
                return self.gen_func(self.bi_out(text_out, tts_out))
            elif self.gen_type == 2:
                return self.gen_func((text_out+tts_out)/2)
            elif self.training:
                return self.gen_func(text_out), self.gen_func(tts_out)
        
        return self.gen_func(text_out)


    

    def find_eos(self, inputs):
        #import pdb; pdb.set_trace()
        return inputs.eq(self.eos)# if self.stop is None else self._stops

class MultiTextGenerator(TextGenerator):
    """
    A single layer of the text output.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """
    def __init__(self, model_opt, field, decoder):
        super(MultiTextGenerator, self).__init__(model_opt, field, decoder)
        self.gen_type = model_opt.gen_type
        if self.gen_type in {1,2,3}:
            self.tts_embed_weight = torch.load(model_opt.add_word_vecs_dec)['model']['encoder.embeddings.embed.make_embedding.emb_luts.0.weight']
        if self.gen_type in {3,4}:
            self.bi_out = nn.Bilinear(len(field), len(field), len(field))
        if self.gen_type == 4:
            self.extra_out = nn.Linear(model_opt.dec_rnn_size, len(field))

    def forward(self, inputs):
        text_out = super().forward(inputs)
        if self.gen_type > 0:
            if self.gen_type == 4:
                return self.gen_func(self.bi_out(text_out, self.extra_out(inputs)))
            tts_out = F.linear(inputs, self.tts_embed_weight)
            if self.gen_type == 3:
                return self.gen_func(self.bi_out(text_out, tts_out))
            elif self.gen_type == 2:
                return self.gen_func((text_out+tts_out)/2)
            elif self.training:
                return self.gen_func(text_out), self.gen_func(tts_out)
        return self.gen_func(text_out)


class EmphaTextGenerator(TextGenerator):
    """
    A single layer of the text output.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """
    def __init__(self, model_opt, field, decoder):
        super(EmphaTextGenerator, self).__init__(model_opt, field, decoder)
        self.emphasis_out = nn.Linear(model_opt.dec_rnn_size, 5)

    def forward(self, inputs):
        text_out = super().forward(inputs)
        emph_out = self.emphasis_out(inputs)
        return self.gen_func(text_out),self.gen_func(emph_out)

class SpeakerGenerator(nn.Module):
    def __init__(self, rnn_type, hidden_size, dropout=0.0,filed=None):
        super(SpeakerGenerator, self).__init__()
        num_directions = 2 
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=hidden_size*num_directions,
                        hidden_size=hidden_size,
                        num_layers=1,
                        dropout=dropout,
                        bidirectional=True)
        self.out = nn.Bilinear(hidden_size,hidden_size, len(filed))
        self.gen = nn.Sequential(
                Cast(torch.float32),
                nn.LogSoftmax(dim=-1)
            )

    @classmethod
    def from_opt(cls, opt,filed):
        """Alternate constructor."""
        return cls(
            'GRU',
            opt.enc_rnn_size,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            filed
            )

    def forward(self, packed_emb, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        #import pdb; pdb.set_trace()
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths_list = lengths.view(-1).tolist()
            packed_emb = pack(packed_emb, lengths_list)
        _, encoder_final = self.rnn(packed_emb)
        
        return self.gen(self.out(encoder_final[0],encoder_final[1]))

class AdversarialGenerator(SpeakerGenerator):
    def __init__(self, rnn_type, hidden_size, dropout=0.0,filed=None):
        super(AdversarialGenerator, self).__init__(rnn_type, hidden_size, dropout,filed)
        self.gen = nn.Sequential(
                nn.Sigmoid()
            )