import torch
import torch.nn as nn
from onmt.modules import PostConvNet, Conv, CBHG, Linear
import onmt.inputters
from onmt.modules import *
from onmt.inputters.audio_dataset import stream_sizes
import os

class SpeechGenerator(nn.Module):
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
        super(SpeechGenerator, self).__init__()
        self.linear  = nn.Linear(model_opt.rnn_size, len(field))
        self.postconv= PostConvNet(80, model_opt.rnn_size)
        self.stop_linear = nn.Linear(model_opt.rnn_size, 1)
        self.has_vocoder=False
        if os.path.exists(model_opt.add_vocoder):
            self.vocoder=MagGenerater(model_opt.rnn_size)
            self.vocoder.load_state_dict(torch.load(model_opt.add_vocoder)['model']['decoder'])
            self.has_vocoder=True
    
    def gen_mag(self,mel):
        if self.has_vocoder:
            return self.vocoder(mel)
        else:
            return None
            
    def forward(self, inputs):
        #import pdb; pdb.set_trace()
        l, b, d=inputs.size()
        mel = self.linear(inputs)
        post_mel = mel.view(-1,b,80) + self.postconv(mel.view(-1,b,80).transpose(1,2)).transpose(1,2)

        stop = self.stop_linear(inputs)
        return mel, post_mel, torch.sigmoid(stop)
    
    def find_eos(self, inputs):
    
        return inputs.eq(self.eos)
    

