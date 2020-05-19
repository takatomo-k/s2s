"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn

from onmt.decoders.decoder import DecoderBase
from onmt.modules import Conv, CBHG
from onmt.utils import GaussianNoise

class PostDecoder(DecoderBase):
    """The Transformer decoder from "Attention is All You Need".
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       copy_attn (bool): if using a separate copy attention
       self_attn_type (str): type of self-attention scaled-dot, average
       dropout (float): dropout parameters
       embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
    """

    def __init__(self, num_mels, hidden_size, num_mags):
        super(PostDecoder, self).__init__()

        self.pre_projection = Conv(num_mels, hidden_size)
        self.cbhg = CBHG(hidden_size)
        self.post_projection = Conv(hidden_size, num_mags)


    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            80,
            opt.dec_rnn_size,
            1025)

    def init_state(self, src, memory_bank, enc_hidden):
        """Initialize decoder state."""
        pass

    def map_state(self, fn):
        pass

    def detach_state(self):
        pass

    def forward(self, tgt, memory_bank, step=None, **kwargs):
        mel = memory_bank.transpose(1, 2)
        mel = self.pre_projection(mel)
        mel = self.cbhg(mel).transpose(1, 2)
        mag_pred = self.post_projection(mel).transpose(1, 2)
        # TODO change the way attns is returned dict => list or tuple (onnx)
        return mag_pred, None

    def _init_cache(self, memory_bank):
        pass
        
    def update_dropout(self, dropout):
        pass